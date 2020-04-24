import json
import logging

import numpy as np
from allennlp.data.fields import ArrayField
from allennlp.data.fields.text_field import TextField

from allennlp.data.fields.metadata_field import MetadataField

from allennlp.data.fields.index_field import IndexField
from typing import Dict, List, Tuple, Optional, Any

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data import Field
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

from docqa.allennlp_custom.data.feature_extractors.feature_extractor import TokenWiseInteractionFeatureExtractor
from docqa.data.dataset_readers.common_utils import combine_sentences_parse, get_span_to_crop_passage_with_answer
from docqa.data.processing.text_semantic_graph import fix_parse_for_items, add_srl_to_sentences

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def crop_semantic_views(semantic_views, crop_start, crop_end, axis=-1, copy=False):
    """
    Crops a token-wise (NxN) features according to cropping range
    :param semantic_views: Semantic views features
    :param crop_start: Crop range start
    :param crop_end: Crop range end
    :return:
    """

    def crop_single_view(arr, crop_start, crop_end, axis=-1, copy=False):
        if len(arr.shape) == 1:
            cropped = arr[crop_start:crop_end]
            if copy:
                return np.copy(cropped)
            else:
                return cropped
        else:
            if axis == -1:
                if len(arr.shape) > 2:
                    cropped = arr[crop_start:crop_end, crop_start:crop_end, :]
                else:
                    cropped = arr[crop_start:crop_end, :]
                if copy:
                    return np.copy(cropped)
                else:
                    return cropped

            elif axis == 0:
                if len(arr.shape) > 2:
                    cropped = arr[:, crop_start:crop_end, crop_start:crop_end]
                else:
                    cropped = arr[:, crop_start:crop_end]

                if copy:
                    return np.copy(cropped)
                else:
                    return cropped
            else:
                raise ValueError("Unsupported crop axis `{0}`".format(axis))

    if isinstance(semantic_views, tuple):
        return [crop_single_view(x, crop_start, crop_end, axis=axis, copy=copy) for x in semantic_views]
    elif isinstance(semantic_views, Dict):
        return {k: crop_single_view(v, crop_start, crop_end, axis=axis, copy=copy) for k, v in semantic_views.items()}
    elif isinstance(semantic_views, np.ndarray):
        return crop_single_view(semantic_views, crop_start, crop_end, axis=axis, copy=copy)
    else:
        raise ValueError("Unexpected type!")


@DatasetReader.register("narrativeqa_summary_with_questions_semantic_optimized_multiview")
class NarrativeQASummaryWithQuestionsSemanticsReaderOptimizedMultiView(DatasetReader):
    """
    Reads a JSON-formatted SQuAD file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``.  We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_texts']`` and
    ``metadata['token_offsets']``.  This is so that we can more easily use the official SQuAD
    evaluation script to get metrics.

    We also support limiting the maximum length for both passage and question. However, some gold
    answer spans may exceed the maximum passage length, which will cause error in making instances.
    We simply skip these spans to avoid errors. If all of the gold answer spans of an example
    are skipped, during training, we will skip this example. During validating or testing, since
    we cannot skip examples, we use the last token as the pseudo gold answer span instead. The
    computed loss will not be accurate as a result. But this will not affect the answer evaluation,
    because we keep all the original gold answer texts.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    lazy : ``bool``, optional (default=False)
        If this is true, ``instances()`` will return an object whose ``__iter__`` method
        reloads the dataset each time it's called. Otherwise, ``instances()`` returns a list.
    passage_length_limit : ``int``, optional (default=None)
        if specified, we will cut the passage if the length of passage exceeds this limit.
    question_length_limit : ``int``, optional (default=None)
        if specified, we will cut the question if the length of passage exceeds this limit.
    skip_invalid_examples: ``bool``, optional (default=False)
        if this is true, we will skip those invalid examples
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 passage_length_limit: int = None,
                 question_length_limit: int = None,
                 fit_answer_in_the_passage_limit: bool = False,
                 skip_invalid_examples: bool = False,
                 semantic_views_extractor: TokenWiseInteractionFeatureExtractor = None,
                 dump_to_file: bool = False,
                 load_partial_data: bool = False,
                 question_id_mod=0,
                 paragraph_id_mod=0,
                 ) -> None:
        super().__init__(lazy)

        self.load_partial_data = load_partial_data
        self.question_id_mod = question_id_mod
        self.paragraph_id_mod = paragraph_id_mod

        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._semantic_views_extractor = semantic_views_extractor
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.skip_invalid_examples = skip_invalid_examples
        self.fit_answer_in_the_passage_limit = fit_answer_in_the_passage_limit
        self.views_axis = 0
        self.dump_to_file = dump_to_file

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            logger.info("Reading the dataset")
            skipped_instances = 0

            q_id_in_file = 0
            for line_id, line in enumerate(dataset_file):
                # partial data
                if self.load_partial_data:
                    if self.paragraph_id_mod > 0:
                        if line_id % self.paragraph_id_mod != 0:
                            continue

                paragraph_json = json.loads(line.strip())

                # document id
                par_line = "P{0:05}".format(line_id)
                doc_id = paragraph_json.get("document_id", par_line)

                # this is used for fixing problems with different tokenization for SRL and spacy + coref
                fix_parse_for_items([paragraph_json], "summary_parse")

                paragraph_parse = combine_sentences_parse(paragraph_json["summary_parse"]["sentences"])
                paragraph_text = paragraph_json["summary"]
                tokenized_paragraph = self._tokenizer.tokenize(paragraph_parse)

                paragraph_semantic_views = None
                if self._semantic_views_extractor is not None:
                    paragraph_semantic_views = self._semantic_views_extractor.extract_features(paragraph_json["summary_parse"])

                for q_id, question_answer in enumerate(paragraph_json['questions']):
                    q_id_in_file += 1
                    # partial data
                    if self.load_partial_data:
                        if self.question_id_mod > 0:
                            if q_id_in_file % self.question_id_mod != 0:
                                continue

                    doc_question_id = "{0}##{1:03}".format(doc_id, q_id)
                    question_text = question_answer["question"].strip().replace("\n", "")

                    question_parse = combine_sentences_parse(question_answer["question_parse"]["sentences"])

                    question_semantic_views = None
                    if self._semantic_views_extractor is not None:
                        if not "srl" in question_answer["question_parse"]["sentences"][0]:
                            add_srl_to_sentences(question_answer["question_parse"], question_answer["question_srl"])

                        fix_parse_for_items([question_answer], "question_parse")

                        question_semantic_views = self._semantic_views_extractor.extract_features(question_answer["question_parse"])

                    answer_texts = [answer for answer in [question_answer['answer1'], question_answer['answer2']]]
                    span_tuple = [x["span"] for x in sorted(question_answer['answers_best_chunks'], key=lambda x: x["score"])]

                    instance = None
                    try:
                        instance_meta = {"id": doc_question_id}
                        instance = self.text_to_instance(question_text,
                                                         question_parse,
                                                         paragraph_text,
                                                         paragraph_parse,
                                                         span_tuple,
                                                         answer_texts,
                                                         tokenized_paragraph,
                                                         paragraph_semantic_views,
                                                         question_semantic_views,
                                                         metadata=instance_meta
                                                         )



                    except Exception as e:
                        logging.error("Exception in file {0}\n"
                                     "Skip: Passage {1}, Question: {2} : {3}".format(file_path, line_id, q_id, e))
                        raise e

                    if instance is not None:
                        yield instance
                    else:
                        skipped_instances += 1

            logging.info("Skipped instances: {0}".format(skipped_instances))

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         question_parse: Dict[str, Any],
                         passage_text: str,
                         passage_parse: Dict[str, Any],
                         token_spans: List[Tuple[int, int]] = None,
                         answer_texts: List[str] = None,
                         passage_tokens: List[Token] = None,
                         paragraph_semantic_views_in=None,
                         question_semantic_views_in=None,
                         metadata: Dict[str, Any] = None
                         ) -> Optional[Instance]:

        # pylint: disable=arguments-differ
        paragraph_semantic_views = paragraph_semantic_views_in
        question_semantic_views = question_semantic_views_in

        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_parse)

        question_tokens = self._tokenizer.tokenize(question_parse)

        if self.passage_length_limit is not None and self.passage_length_limit < len(passage_tokens):
            answer_start, answer_end = tuple(token_spans[0])
            crop_start = 0
            crop_end = self.passage_length_limit
            if self.passage_length_limit < answer_end:
                if self.fit_answer_in_the_passage_limit:
                    curr_crop_start, curr_crop_end, new_answ_start, new_answ_end = get_span_to_crop_passage_with_answer(
                                                                                                    len(passage_tokens),
                                                                                                    self.passage_length_limit,
                                                                                                    answer_start,
                                                                                                    answer_end)

                    if curr_crop_start is not None \
                            and new_answ_start >= 0:
                        crop_start = curr_crop_start
                        crop_end = curr_crop_end
                        answer_start = new_answ_start
                        answer_end = new_answ_end
                        #logging.info("Cropped!")
                    else:
                        #logging.info("Skipped!")
                        return None
                else:
                    #logging.info("Skipped!")
                    return None

            # crop happens here
            passage_tokens = passage_tokens[crop_start: crop_end]
            token_spans = [[answer_start, answer_end]]
            paragraph_semantic_views = crop_semantic_views(paragraph_semantic_views, crop_start, crop_end,
                                                           self.views_axis, copy=False)

            assert(token_spans[0][0] >= 0)
            assert(token_spans[0][1] - 1 < len(passage_tokens))


        if self.question_length_limit is not None:
            question_tokens = question_tokens[: self.question_length_limit]
            question_semantic_views = crop_semantic_views(question_semantic_views, 0, self.question_length_limit,
                                                          self.views_axis, copy=False)

        # if self.dump_to_file:
        #     dump_item = {"question_tokens": question_tokens,
        #                  "passage_tokens": passage_tokens,
        #                  "token_spans": token_spans,
        #                  "answer_texts": answer_texts,
        #                  "paragraph_semantic_views": np.copy(paragraph_semantic_views),
        #                  "question_semantic_views": np.copy(question_semantic_views)
        #                  }
        #     self._dump_file.write

        return self.make_reading_comprehension_instance_with_semantics(question_tokens,
                                                        passage_tokens,
                                                        self._token_indexers,
                                                        passage_text,
                                                        token_spans,
                                                        answer_texts,
                                                        additional_metadata=metadata,
                                                        paragraph_semantic_views=paragraph_semantic_views,
                                                        question_semantic_views=question_semantic_views)


    def make_reading_comprehension_instance_with_semantics(self, question_tokens: List[Token],
                                            passage_tokens: List[Token],
                                            token_indexers: Dict[str, TokenIndexer],
                                            passage_text: str,
                                            token_spans: List[Tuple[int, int]] = None,
                                            answer_texts: List[str] = None,
                                            additional_metadata: Dict[str, Any] = None,
                                            paragraph_semantic_views=None,
                                            question_semantic_views=None
                                            ) -> Instance:
        """
        Converts a question, a passage, and an optional answer (or answers) to an ``Instance`` for use
        in a reading comprehension model.

        Creates an ``Instance`` with at least these fields: ``question`` and ``passage``, both
        ``TextFields``; and ``metadata``, a ``MetadataField``.  Additionally, if both ``answer_texts``
        and ``char_span_starts`` are given, the ``Instance`` has ``span_start`` and ``span_end``
        fields, which are both ``IndexFields``.

        Parameters
        ----------
        question_tokens : ``List[Token]``
            An already-tokenized question.
        passage_tokens : ``List[Token]``
            An already-tokenized passage that contains the answer to the given question.
        token_indexers : ``Dict[str, TokenIndexer]``
            Determines how the question and passage ``TextFields`` will be converted into tensors that
            get input to a model.  See :class:`TokenIndexer`.
        passage_text : ``str``
            The original passage text.  We need this so that we can recover the actual span from the
            original passage that the model predicts as the answer to the question.  This is used in
            official evaluation scripts.
        token_spans : ``List[Tuple[int, int]]``, optional
            Indices into ``passage_tokens`` to use as the answer to the question for training.  This is
            a list because there might be several possible correct answer spans in the passage.
            Currently, we just select the most frequent span in this list (i.e., SQuAD has multiple
            annotations on the dev set; this will select the span that the most annotators gave as
            correct).
        answer_texts : ``List[str]``, optional
            All valid answer strings for the given question.  In SQuAD, e.g., the training set has
            exactly one answer per question, but the dev and test sets have several.  TriviaQA has many
            possible answers, which are the aliases for the known correct entity.  This is put into the
            metadata for use with official evaluation scripts, but not used anywhere else.
        additional_metadata : ``Dict[str, Any]``, optional
            The constructed ``metadata`` field will by default contain ``original_passage``,
            ``token_offsets``, ``question_tokens``, ``passage_tokens``, and ``answer_texts`` keys.  If
            you want any other metadata to be associated with each instance, you can pass that in here.
            This dictionary will get added to the ``metadata`` dictionary we already construct.
        """
        additional_metadata = additional_metadata or {}
        fields: Dict[str, Field] = {}

        # This is separate so we can reference it later with a known type.
        passage_field = TextField(passage_tokens, token_indexers)
        fields['passage'] = passage_field
        fields['question'] = TextField(question_tokens, token_indexers)
        metadata = {'original_passage': passage_text,
                    'question_tokens': [token.text for token in question_tokens],
                    'passage_tokens': [token.text for token in passage_tokens], }

        if answer_texts:
            metadata['answer_texts'] = answer_texts

        if token_spans:
            metadata["token_spans"] = token_spans

            # assume spans are sorted by some criteria
            span_start = token_spans[0][0]
            span_end = token_spans[0][1] - 1
            assert(span_start <= span_end)
            if span_end > len(passage_tokens) - 1:
                return None

            fields['span_start'] = IndexField(span_start, passage_field)
            fields['span_end'] = IndexField(span_end, passage_field)

        if paragraph_semantic_views is not None \
            and question_semantic_views is not None:
            if isinstance(paragraph_semantic_views, tuple) or isinstance(paragraph_semantic_views, list):
                fields['passage_sem_views_q'] = ArrayField(paragraph_semantic_views[0])
                fields['passage_sem_views_k'] = ArrayField(paragraph_semantic_views[1])
            else:
                logging.info("paragraph_semantic_views type:{0}".format(type(paragraph_semantic_views)))
                fields['passage_sem_views_q'] = ArrayField(paragraph_semantic_views)

            if isinstance(question_semantic_views, tuple) or isinstance(question_semantic_views, list):
                fields['question_sem_views_q'] = ArrayField(question_semantic_views[0])
                fields['question_sem_views_k'] = ArrayField(question_semantic_views[1])
            else:
                logging.info("question_semantic_views type:{0}".format(type(question_semantic_views)))
                fields['question_sem_views_q'] = ArrayField(question_semantic_views)

        metadata.update(additional_metadata)
        fields['metadata'] = MetadataField(metadata)


        return Instance(fields)


