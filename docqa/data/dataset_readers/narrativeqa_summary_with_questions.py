import json
import logging

from allennlp.data.fields.text_field import TextField

from allennlp.data.fields.metadata_field import MetadataField

from allennlp.data.fields.index_field import IndexField
from typing import Dict, List, Tuple, Optional, Any

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data import Field
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from docqa.data.dataset_readers.common_utils import combine_sentences_parse, get_span_to_crop_passage_with_answer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("narrativeqa_summary_with_questions")
class NarrativeQASummaryWithQuestionsReader(DatasetReader):
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
                 skip_invalid_examples: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.skip_invalid_examples = skip_invalid_examples
        self.fit_answer_in_the_passage_limit = fit_answer_in_the_passage_limit

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            logger.info("Reading the dataset")
            skipped_instances = 0
            for line_id, line in enumerate(dataset_file):

                paragraph_json = json.loads(line.strip())

                # document id
                par_line = "P{0:05}".format(line_id)
                doc_id = paragraph_json.get("document_id", par_line)

                paragraph_parse = combine_sentences_parse(paragraph_json["summary_parse"]["sentences"])
                paragraph_text = paragraph_json["summary"]
                tokenized_paragraph = self._tokenizer.tokenize(paragraph_parse)

                for q_id, question_answer in enumerate(paragraph_json['questions']):
                    doc_question_id = "{0}##{1:03}".format(doc_id, q_id)

                    question_text = question_answer["question"].strip().replace("\n", "")
                    question_parse = combine_sentences_parse(question_answer["question_parse"]["sentences"])

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
                                                         metadata=instance_meta
                                                         )

                    except Exception as e:
                        logging.info("Exception in file {0}\n"
                                     "Skip: Passage {1}, Question: {2}".format(file_path, line_id, q_id))

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
                         metadata: Dict[str, Any] = None
                         ) -> Optional[Instance]:
        # pylint: disable=arguments-differ
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_parse)

        question_tokens = self._tokenizer.tokenize(question_parse)

        if self.passage_length_limit is not None and self.passage_length_limit < len(passage_tokens):
            span_start, span_end = tuple(token_spans[0])
            if self.passage_length_limit < span_end:
                if self.fit_answer_in_the_passage_limit:
                    crop_start, crop_end, new_answ_start, new_answ_end = get_span_to_crop_passage_with_answer(
                                                                                                    len(passage_tokens),
                                                                                                    self.passage_length_limit,
                                                                                                    span_start,
                                                                                                    span_end)

                    if crop_start is not None:
                        passage_tokens = passage_tokens[crop_start: crop_end]
                        token_spans = [[new_answ_start, new_answ_end]]
                        logging.info("Cropped")
                    else:
                        return None
                else:
                    return None
            else:
                passage_tokens = passage_tokens[: self.passage_length_limit]

        if self.question_length_limit is not None:
            question_tokens = question_tokens[: self.question_length_limit]

        return self.make_reading_comprehension_instance(question_tokens,
                                                        passage_tokens,
                                                        self._token_indexers,
                                                        passage_text,
                                                        token_spans,
                                                        answer_texts,
                                                        additional_metadata=metadata)

    def make_reading_comprehension_instance(self, question_tokens: List[Token],
                                            passage_tokens: List[Token],
                                            token_indexers: Dict[str, TokenIndexer],
                                            passage_text: str,
                                            token_spans: List[Tuple[int, int]] = None,
                                            answer_texts: List[str] = None,
                                            additional_metadata: Dict[str, Any] = None) -> Instance:
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

        metadata.update(additional_metadata)
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)

