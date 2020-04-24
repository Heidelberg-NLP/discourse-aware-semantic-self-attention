"""
The ``evaluate_custom`` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.

.. code-block:: bash

    $ python -m allennlp.run evaluate_custom --help
    usage: run [command] evaluate_custom [-h] --archive_file ARCHIVE_FILE
                                --evaluation_data_file EVALUATION_DATA_FILE
                                [--cuda_device CUDA_DEVICE]

    Evaluate the specified model + dataset

    optional arguments:
    -h, --help            show this help message and exit
    --archive_file ARCHIVE_FILE
                            path to an archived trained model
    --evaluation_data_file EVALUATION_DATA_FILE
                            path to the file containing the evaluation data
    --output_file OUTPUT_FILE
                            path to optional output file with detailed predictions
    --cuda_device CUDA_DEVICE
                            id of GPU to use (if any)
"""
import os

from typing import Dict, Any, Iterable
import argparse
from contextlib import ExitStack
import json
import logging

import torch
import tqdm

from allennlp.commands.subcommand import Subcommand
from allennlp.common import Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data import Instance
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from torch import Tensor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class EvaluateCustom(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:  # pylint: disable=protected-access
        description = '''Evaluate the specified model + dataset with optional output'''
        subparser = parser.add_parser('evaluate_custom',
                                      description=description,
                                      help='Evaluate the specified model + dataset with optional output')
        subparser.add_argument('--archive_file',
                               type=str,
                               required=True,
                               help='path to an archived trained model')
        subparser.add_argument('--evaluation_data_file',
                               type=str,
                               required=True,
                               help='path to the file containing the evaluation data')
        subparser.add_argument('--output_file',
                               type=str,
                               required=False,
                               help='output file for raw evaluation results')
        subparser.add_argument('--cuda_device',
                               type=int,
                               default=-1,
                               help='id of GPU to use (if any)')
        subparser.add_argument('--batch_size',
                               type=int,
                               default=-1,
                               help='batch_size to use. If -1 the one from configuration is used')
        subparser.add_argument('--start_id',
                               type=int,
                               default=-1,
                               help='Instance start id')
        subparser.add_argument('--end_id',
                               type=int,
                               default=-1,
                               help='Instance start id')
        subparser.add_argument('--file_open_mode',
                               type=str,
                               default="w",
                               help='File writing mode')
        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a HOCON structure used to override the experiment configuration')

        subparser.set_defaults(func=evaluate_from_args)

        return subparser


def evaluate(model: Model,
             instances: Iterable[Instance],
             data_iterator: DataIterator,
             output_file: str = None,
             file_mode="w") -> Dict[str, Any]:
    model.eval()

    iterator = data_iterator(instances, num_epochs=1, shuffle=False)
    logger.info("Iterating over dataset")
    generator_tqdm = Tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))
    with ExitStack() as stack:
        if output_file is None:
            file_handle = None
        else:
            file_handle = stack.enter_context(open(output_file, file_mode))

        for batch in generator_tqdm:
            model_output = model(**batch)
            metrics = model.get_metrics()
            if file_handle:
                id2label = model.vocab.get_index_to_token_vocabulary("labels")
                _persist_data(file_handle, batch.get("metadata"), model_output, id2label=id2label)
            description = ', '.join(["%s: %.2f" % (name, value) for name, value in metrics.items()]) + " ||"
            generator_tqdm.set_description(description)

    return model.get_metrics()


def _persist_data(file_handle, metadata, model_output, id2label=None) -> None:
    if metadata:
        batch_size = len(metadata)
        for index, meta in enumerate(metadata):
            res = {}
            res["id"] = meta.get("id", "n/a")
            res["meta"] = meta
            # We persist model output which matches batch_size in length and is not a Variable
            for key, value in model_output.items():
                curr_value = value
                if isinstance(value, torch.autograd.Variable) or isinstance(value, Tensor):
                    curr_value = value.data.tolist()

                if not isinstance(curr_value, torch.autograd.Variable) \
                        and isinstance(curr_value, list) \
                        and len(curr_value) == batch_size:
                    val = curr_value[index]
                    res[key] = val

            if "label_probs" in res and id2label is not None:
                labels_by_probs = sorted([[id2label[li], lp] for li, lp in enumerate(res["label_probs"])], key=lambda x:x[1], reverse=True)
                res["labels_by_prob"] = labels_by_probs
                res["label_predicted"] = labels_by_probs[0][0]

            file_handle.write(json.dumps(res))
            file_handle.write("\n")


def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info("{0}: {1}".format(arg, getattr(args, arg)))

    # Load from archive
    cuda_device = args.cuda_device

    logging.info("cuda_device:{0}".format(cuda_device))
    archive = load_archive(args.archive_file, cuda_device=cuda_device, overrides=args.overrides)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    # Load the evaluation data
    dataset_reader = DatasetReader.from_params(config.pop('validation_dataset_reader')
                                               if "validation_dataset_reader" in config
                                               else config.pop('dataset_reader'))
    evaluation_data_path = args.evaluation_data_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)

    batch_size = args.batch_size
    start_id = args.start_id
    end_id = args.end_id

    dataset = dataset_reader.read(evaluation_data_path)
    file_mode = args.file_open_mode
    if start_id > 0 or end_id > 0:
        if not isinstance(dataset, list):
            raise ValueError("dataset must be list when start_id and end_id are set")

        start_id = max(start_id, 0)
        if end_id <= 0:
            end_id = len(dataset)

        dataset = dataset[start_id: end_id]

    iterator_config = config.pop('validation_iterator') if "validation_iterator" in config else config.pop('iterator')
    if batch_size > -1:
        if "base_iterator" in iterator_config:
            iterator_config["base_iterator"]["batch_size"] = batch_size
        else:
            iterator_config["batch_size"] = batch_size

    iterator = DataIterator.from_params(iterator_config)

    iterator.index_with(model.vocab)

    metrics = evaluate(model, dataset, iterator, args.output_file, file_mode=file_mode)
    if args.output_file:
        absolute_path = os.path.abspath(args.output_file)
        logging.info("Output saved to \n{}".format(absolute_path))
    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    return metrics
