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


class EvaluateQaNetSemanticFlat(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:  # pylint: disable=protected-access
        description = '''Evaluate the specified model + dataset with optional output'''
        subparser = parser.add_parser('evaluate_qanet_semantic_flat',
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
                               default=1,
                               help='batch_size to use. If -1 the one from configuration is used')
        subparser.add_argument('--item_ids',
                               type=str,
                               required=False,
                               help='list of items to select')
        subparser.add_argument('--output_attention',
                               default="False",
                               type=str,
                               required=False,
                               help='Exports the model attentions of it is supported')
        subparser.add_argument('--display_attention_matplot',
                               default="False",
                               type=str,
                               required=False,
                               help='Displays the attention with matplotlib')
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


from types import SimpleNamespace


def create_argparse_namespace(archive_file: str,
                              evaluation_data_file: str,
                              output_file: str,
                              item_ids:str,
                              cuda_device=-1,
                              batch_size=1,
                              output_attention: str = "False",
                              file_open_mode = "w",
                              overrides: str = "",
                              start_id = -1,
                              end_id = -1,
                              display_attention_matplot: str = "False",
                              ):
    args_ns = SimpleNamespace()

    args_ns.archive_file = archive_file
    args_ns.evaluation_data_file = evaluation_data_file
    args_ns.output_file = output_file
    args_ns.item_ids = item_ids
    args_ns.cuda_device = cuda_device
    args_ns.batch_size = batch_size
    args_ns.output_attention = output_attention
    args_ns.file_open_mode = file_open_mode
    args_ns.overrides = overrides
    args_ns.start_id = start_id
    args_ns.end_id = end_id
    args_ns.display_attention_matplot = display_attention_matplot

    return args_ns


def evaluate(model: Model,
             instances: Iterable[Instance],
             data_iterator: DataIterator,
             output_file: str = None,
             file_mode="w",
             id_to_meta: Dict[str, Any] = {},
             feat_id_to_feat_name: Dict[int, str]= {}
             ) -> Dict[str, Any]:
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
                _persist_data(file_handle, batch.get("metadata"), model_output,
                              id2label=id2label,
                              id_to_meta=id_to_meta,
                              feat_id_to_feat_name=feat_id_to_feat_name)
            description = ', '.join(["%s: %.2f" % (name, value) for name, value in metrics.items()]) + " ||"
            generator_tqdm.set_description(description)

    return model.get_metrics()

#
# def crop_values(curr_value, crop_start, crop_end):
#     if len(curr_value.shape) == 1:
#         if curr_value.shape
#         curr_value = curr_value[crop_start:crop_end]
#     elif len(curr_value.shape) == 2:
#         curr_value = curr_value[crop_start:crop_end, crop_start:crop_end]
#     elif len(curr_value.shape) == 3:
#         # this is att_heads x seq x seq
#         curr_value = curr_value[:, crop_start:crop_end, crop_start:crop_end]
#     else:
#         raise ValueError("Array with shape {0} is not supported!".format(len(curr_value.shape)))
#
#     return curr_value

def attentions_to_json(attentions_metadata, index, batch_size, sequence_len, crop_range, res_type="np.array"):
    res = {}
    if isinstance(attentions_metadata, dict):
        for key, value in attentions_metadata.items():
            if value is None:
                res[key] = value
            elif isinstance(value, torch.autograd.Variable) or isinstance(value, Tensor):
                if value.data.shape[0] == batch_size:
                    curr_value = value.data[index].cpu().numpy()
                else:
                    curr_value = value.data.cpu().numpy()

                if res_type == "list":
                    res[key] = curr_value
                else:
                    res[key] = curr_value.tolist()
            elif isinstance(value, dict):
                res[key] = attentions_to_json(value, index, batch_size, sequence_len, crop_range)
            elif isinstance(value, list):
                res[key] = attentions_to_json(value, index, batch_size, sequence_len, crop_range)
            else:
                raise NotImplementedError("This case is not supported: isinstance(value, {0})".format(type(value)))
        return res
    elif isinstance(attentions_metadata, list):
        res = []
        for value in attentions_metadata:
            val = attentions_to_json(value, index, batch_size, sequence_len, crop_range)
            res.append(val)
        return res

    return res


def _persist_data(file_handle, metadata, model_output, id2label=None,
                  id_to_meta=None,
                  feat_id_to_feat_name=None,
                  display_attention_matplot=False) -> None:
    if metadata:
        batch_size = len(metadata)
        for index, meta in enumerate(metadata):
            res = {}
            item_id = meta.get("id", "n/a")
            res["id"] = item_id
            res["meta"] = meta
            # We persist model output which matches batch_size in length and is not a Variable
            for key, value in model_output.items():
                if key == "output_metadata":
                    crop_range = None
                    if id_to_meta is not None:
                        crop_range = id_to_meta.get(item_id, None)

                    sequence_len = model_output["span_start_logits"].shape[-1]
                    attentions_metadata = attentions_to_json(value, index, batch_size, sequence_len , crop_range)
                    #res["attentions_metadata"] = attentions_metadata
                    #print(attentions_metadata)


                curr_value = value
                if isinstance(value, torch.autograd.Variable) or isinstance(value, Tensor):
                    curr_value = value.data.tolist()

                if key in ["best_span_semantic_features", "best_span_tokens"]:
                    res[key] = curr_value
                elif not isinstance(curr_value, torch.autograd.Variable) \
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


def evaluate_from_args(args: argparse.Namespace, func_eval=None) -> Dict[str, Any]:
    # USAGE:
    # docqa/run.py
    # evaluate_custom
    # --archive_file
    # _trained_models/qanet_semantic_flat_concat_sdp_debug/model.tar.gz
    # --evaluation_data_file
    # /Users/mihaylov/research/document-parsing-pipeline/tests/fixtures/data/narrativeqa/third_party/wikipedia/summaries-all.csv.parsed.jsonl.srl.jsonl.with_q_spans.jsonl.with_exp.with_sdp.json.train.2
    # --output_file
    # predictions_dev.json
    # --batch_size=1
    # --item_ids
    # "00936497f5884881f1df23f4834f6739552cee8b##016[15:30];00936497f5884881f1df23f4834f6739552cee8b##005[17:45];0029bdbe75423337b551e42bb31f9a102785376f##023"
    # --output_attention
    # True

    if func_eval is None:
        func_eval = evaluate

    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info("{0}: {1}".format(arg, getattr(args, arg)))

    # Load from archive
    cuda_device = args.cuda_device
    output_attention = args.output_attention.lower() == "true"
    display_attention_matplot = args.display_attention_matplot.lower() == "true"

    # selected ids to validate
    item_ids = []
    item_ids_with_range = {}
    if args.item_ids:
        separator = ";"
        item_ids_str = args.item_ids
        item_ids = item_ids_str.split(separator)
        for item_id in item_ids:
            if "[" in item_id and "]" in item_id:
                tokens_range = [int(x) for x in item_id.split("[")[-1].replace("]", "").split(":")]
                item_id_only = item_id.split("[")[0]
                item_ids_with_range[item_id_only] = {"attention_range": tokens_range}
            else:
                item_ids_with_range[item_id] = None

        item_ids = set(item_ids_with_range.keys())

    logging.info("cuda_device:{0}".format(cuda_device))
    archive = load_archive(args.archive_file, cuda_device=cuda_device, overrides=args.overrides)
    config = archive.config
    prepare_environment(config)

    model = archive.model
    model.eval()

    if output_attention:
        if hasattr(model, "return_output_metadata"):
            model.return_output_metadata = output_attention
        else:
            raise Exception("Model {0} does not support output of the attention weights!".format(model))

    # Load the evaluation data
    dataset_reader = DatasetReader.from_params(config.pop('validation_dataset_reader')
                                               if "validation_dataset_reader" in config
                                               else config.pop('dataset_reader'))

    feat_id_to_feat_name = {}
    if hasattr(dataset_reader, "_semantic_views_extractor"):
        feat_id_to_feat_name = dataset_reader._semantic_views_extractor.get_vocab_feats_id2name()

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

    selected_dataset = []
    if len(item_ids) > 0:
        for item in dataset:
            item_id = item.fields["metadata"]["id"]
            if item_id in item_ids:
                selected_dataset.append(item)
            else:
                del item

        dataset = selected_dataset

    iterator_config = config.pop('validation_iterator') if "validation_iterator" in config else config.pop('iterator')
    if batch_size > -1:
        if "base_iterator" in iterator_config:
            iterator_config["base_iterator"]["batch_size"] = batch_size
        else:
            iterator_config["batch_size"] = batch_size

    iterator = DataIterator.from_params(iterator_config)

    iterator.index_with(model.vocab)

    metrics = func_eval(model, dataset, iterator, args.output_file,
                       file_mode=file_mode,
                       id_to_meta=item_ids_with_range,
                       feat_id_to_feat_name=feat_id_to_feat_name)

    if args.output_file:
        absolute_path = os.path.abspath(args.output_file)
        logging.info("Output saved to \n{}".format(absolute_path))
        with open(absolute_path + ".id2featname", mode="w") as fp:
            json.dump(feat_id_to_feat_name, fp)


    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    return metrics
