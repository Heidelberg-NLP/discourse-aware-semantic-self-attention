import random
from copy import deepcopy

import numpy as np
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Token
from typing import Dict, List, Any, Iterator
import json
import logging

from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, ListField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

from docqa.allennlp_custom import ReadSentenceParseTypeWise
from docqa.data.dataset_readers.common_utils import token_indexer_dict_from_params, any_in_set, \
    combine_parse_fields_if_both_exists_and_add_new_field, spacy_entities_mapping, extract_and_map_entities, \
    combine_sentences_parse
from docqa.data.dataset_readers.narrativeqa_reader import NarrativeQaReader
from docqa.utils.processing_utils import load_json_list, get_token_lookup_pointers

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


try:
    from spacy.lang.en.stop_words import STOP_WORDS
except:
    from spacy.en import STOP_WORDS

def get_range_from_parse(parse, span, fields_to_crop_with_span, fields_to_take_full=None):
    new_parse = {}

    span_start = span[0]
    span_end = span[1]

    if fields_to_crop_with_span is None and fields_to_take_full is None:
        raise ValueError("fields_to_crop_with_span and fields_to_take_full are both None!")

    # fields to take span from
    if fields_to_crop_with_span is not None:
        for field in fields_to_crop_with_span:
            new_parse[field] = parse[field][span_start:span_end]

    # fields to copy completely
    if fields_to_take_full is not None:
        for field in fields_to_take_full:
            new_parse[field] = deepcopy(parse[field])

    return new_parse



def get_new_parse_token(token, lemma="", pos="", ent="O", ent_iob="O"):
    single_token_parse = {
        "tokens": [token],
        "pos": [pos],
        "lemmas": [lemma],
        "ent": [ent],
        "ent_iob": [ent_iob]
    }

    return single_token_parse

STORY_START_TOKEN = "@STORYSTART@"
STORY_SKIP_TOKEN = "@SKIP@"

def get_retrieved_chunks_parse_and_concat(question_with_ranking,
                                          text_search_chunk_splits,
                                          chunks_parsed,
                                          top_retrieved_chunks,
                                          fields_to_take_full,
                                          fields_to_crop_with_span
                                          ):

    ranked_chunks_meta = question_with_ranking["text_ranked_chunks"][:top_retrieved_chunks]
    context_chunks_parse = []

    # rank chronologically
    ranked_chunks_meta = sorted(ranked_chunks_meta, key=lambda x: x[0])
    for ch_id, chunk_meta in enumerate(ranked_chunks_meta):
        if ch_id == 0 and chunk_meta[0] > 1:
            context_chunks_parse.append(get_new_parse_token(STORY_START_TOKEN))

        split_meta = text_search_chunk_splits[chunk_meta[0]]
        chunk_parse = chunks_parsed[split_meta["chunk_id"]]["parse"]
        tokens_span = split_meta["tokens_span"]

        chunk_parse_tokens = get_range_from_parse(chunk_parse, tokens_span,
                                                  fields_to_crop_with_span, fields_to_take_full)

        context_chunks_parse.append(chunk_parse_tokens)
        context_chunks_parse.append(get_new_parse_token(STORY_SKIP_TOKEN))

    combined_chunk_parses = combine_sentences_parse(context_chunks_parse)

    return combined_chunk_parses

TARGET_HANDLE_ALL = "all"
TARGET_HANDLE_RANDOM = "random"
TARGET_HANDLE_ANSWER1 = "answer1"
TARGET_HANDLE_ANSWER2 = "answer2"

TARGET_HANDLE_LIST = [TARGET_HANDLE_ALL, TARGET_HANDLE_RANDOM, TARGET_HANDLE_ANSWER1, TARGET_HANDLE_ANSWER2]


@DatasetReader.register("narrativeqa_context_and_questions_seq2seq_parse")
class NarrativeQAContextAndQuestionsSeq2SeqParseReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``SimpleSeq2Seq`` model, or any model with a matching API.
    Expected format for each input line: <source_sequence_string>\t<target_sequence_string>
    The output of ``read`` is a list of ``Instance`` s with the fields:
        source_tokens: ``TextField`` and
        target_tokens: ``TextField``
    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.
    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    """

    def __init__(self,
                 dataset_dir: str,
                 dataset_processed_data_dir: str,
                 processed_questions_file: str,
                 question_to_context_ranking_file: str,
                 top_retrieved_chunks: int,
                 source_context_type: str = "ranked_chunks",  # "summary"
                 processed_summaries_file: str = None,
                 doc_ids: List[str] = None,
                 context_tokenizer: Tokenizer = None,
                 question_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 context_token_indexers: Dict[str, TokenIndexer] = None,
                 question_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 return_context_tokens_pointers: bool = True,
                 handle_multiple_target_in_train: str = "all",
                 lazy: bool = False) -> None:
        super().__init__(lazy)

        if source_context_type not in ["summary", "ranked_chunks"]:
            raise Exception("`source_context_type` should be either `summary` or `ranked_chunks`")
        self._source_context_type = source_context_type

        if source_context_type == "summary" and processed_summaries_file is None:
            raise Exception("When `source_context_type` is 'summary',  `processed_summaries_file` should be specified")
        self._processed_summaries_file = processed_summaries_file

        if handle_multiple_target_in_train not in TARGET_HANDLE_LIST:
            raise ValueError("handle_multiple_target_in_train value should be `all` or `random`")

        self._handle_multiple_target_in_train = handle_multiple_target_in_train

        self._dataset_dir = dataset_dir
        self._dataset_processed_data_dir = dataset_processed_data_dir
        self._processed_questions_file = processed_questions_file
        self._question_to_context_ranking_file = question_to_context_ranking_file
        self._top_retrieved_chunks = top_retrieved_chunks

        self._doc_ids_to_use = None if doc_ids is None or len(doc_ids) == 0 else doc_ids

        self._context_tokenizer = context_tokenizer or WordTokenizer()
        self._question_tokenizer = question_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._context_tokenizer
        self._context_token_indexers = context_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._question_token_indexers = question_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._context_token_indexers
        self._source_add_start_token = source_add_start_token
        self._return_context_tokens_pointers = return_context_tokens_pointers


    @overrides
    def _read(self, file_path):
        # "dataset_dir": "/Users/mihaylov/research/document-parsing-pipeline/test/fixtures/data/narrativeqa",
        # "dataset_processed_data_dir": "/Users/mihaylov/research/document-parsing-pipeline/test/fixtures/data/narrativeqa/processed_data",
        # "processed_questions_file": "/Users/mihaylov/research/document-parsing-pipeline/test/fixtures/data/narrativeqa/processed_data/qaps_all.csv.parsed.jsonl.srl.jsonl",
        # "question_to_context_ranking_file": "/Users/mihaylov/research/document-parsing-pipeline/test/fixtures/data/narrativeqa/processed_data/ranked_q2p_chunk200_ngram12_lemmas-5docs.jsonl",

        # In this method file_path is actually expected to by the name of the set train/val/test
        allowed_values = ["train", "valid", "test"]

        set_value_delim = "," if "," in file_path else ";"
        sets_to_load = [xx for xx in [x.strip() for x in file_path.split(set_value_delim)] if len(xx) > 0]

        # validate that the sets are allowed
        for set_name in sets_to_load:
            if set_name not in allowed_values:
                error_msg = "file_path must be a string with comma separated`train` or `valid` or `test`  but it is {0} instead!".format(file_path)
                logging.error(error_msg)
                raise ValueError(error_msg)

        # doc_ids
        doc_ids_to_read = self._doc_ids_to_use

        # load documents and questions
        reader = NarrativeQaReader(self._dataset_dir)
        documents_meta_list = reader.load_documents_meta_from_csv(doc_ids=doc_ids_to_read, set_names=sets_to_load)
        if self._processed_questions_file is not None:
            questions_list = reader.load_json_list(self._processed_questions_file,
                                                   doc_ids=doc_ids_to_read, set_names=sets_to_load)
        else:
            questions_list = reader.load_questions_from_csv(doc_ids=doc_ids_to_read, set_names=sets_to_load)
        #summaries_list = reader.load_summaries_from_csv(doc_ids=doc_ids_to_read, set_names=sets_to_load)
        #story_content_file_format = reader.story_content_file_format

        # doc_ids_to_read_list=[]
        if doc_ids_to_read is None or len(doc_ids_to_read) == 0:
            doc_ids_to_read = [x["document_id"] for x in documents_meta_list]

        # debug info
        logger.info("{0} documents".format(len(documents_meta_list)))
        logger.info("{0} questions".format(len(questions_list)))
        #logger.info("{0} summaries".format(len(summaries_list)))

        class DocumentWithChunksItem(object):

            def __init__(self, doc_id,
                         doc_with_ranking,
                         top_retrieved_chunks,
                         dataset_processed_data_dir):
                self.doc_id = doc_id
                self.data = doc_with_ranking
                self._top_retrieved_chunks = top_retrieved_chunks
                self._chunks_file = "{0}/{1}.content.parsed.chunks.jsonl".format(dataset_processed_data_dir, doc_id)
                self._chunks_parsed = None

            def get_context_parse(self,
                                  q_id,
                                  fields_to_take_full,
                                  fields_to_crop_with_span
                                  ):
                question_with_ranking = self.get_question_item(q_id)
                text_search_chunk_splits = self.data["text_search_chunks"]

                if self._chunks_parsed is None:
                    error_chunks_parsed = ""
                    try:
                        chunks_parsed = load_json_list(self._chunks_file)
                        self._chunks_parsed = chunks_parsed
                    except Exception as e:
                        chunks_parsed = []
                        error_chunks_parsed = str(e)
                        logging.exception(e)
                        raise e

                return get_retrieved_chunks_parse_and_concat(question_with_ranking,
                                                      text_search_chunk_splits,
                                                      self._chunks_parsed,
                                                      self._top_retrieved_chunks,
                                                      fields_to_take_full,
                                                      fields_to_crop_with_span)

            def get_question_item(self, q_id):
                return self.data["questions_with_ranks"][q_id]

        class DocumentsWithParsedChunksIterator(object):
            def __init__(self,
                         question_to_context_ranking_file,
                         doc_ids,
                         set_names,
                         top_retrieved_chunks,
                         dataset_processed_data_dir):
                self._question_to_context_ranking_file = question_to_context_ranking_file
                self._doc_ids_to_read = doc_ids
                self._sets_to_load = set_names
                self._top_retrieved_chunks = top_retrieved_chunks
                self._dataset_processed_data_dir = dataset_processed_data_dir

            def __iter__(self):
                for doc_with_ranking in reader.load_json_list(self._question_to_context_ranking_file,
                                      doc_ids=self._doc_ids_to_read, set_names=self._sets_to_load):
                    item = DocumentWithChunksItem(doc_id=doc_with_ranking["document_id"],
                                                  doc_with_ranking=doc_with_ranking,
                                                  top_retrieved_chunks=self._top_retrieved_chunks,
                                                  dataset_processed_data_dir=self._dataset_processed_data_dir)

                    yield item

        class DocumentSummaryItem(object):

            def __init__(self, doc_id,
                         data):
                self.doc_id = doc_id
                self.data = data

            def get_context_parse(self,
                                  q_id,
                                  fields_to_take_full,
                                  fields_to_crop_with_span
                                  ):
                context_sequence = combine_sentences_parse(self.data["summary_parse"]["sentences"])

                return context_sequence

            def get_question_item(self, q_id):
                return None

        class DocumentsSummariesIterator(object):
            def __init__(self,
                         parsed_summaries_file,
                         doc_ids,
                         set_names):
                self._parsed_summaries_file = parsed_summaries_file
                self._doc_ids_to_read = doc_ids
                self._sets_to_load = set_names

            def __iter__(self):
                for doc_summary in reader.load_json_list(self._parsed_summaries_file,
                                      doc_ids=self._doc_ids_to_read, set_names=self._sets_to_load):
                    item = DocumentSummaryItem(doc_id=doc_summary["document_id"],
                                               data=doc_summary)

                    yield item

        doc_iterator = None
        if self._source_context_type == "summary":
            doc_iterator = DocumentsSummariesIterator(self._processed_summaries_file,
                                              doc_ids=doc_ids_to_read,
                                              set_names=sets_to_load)
        else:
            doc_iterator = DocumentsWithParsedChunksIterator(self._question_to_context_ranking_file,
                                              doc_ids=doc_ids_to_read,
                                              set_names=sets_to_load,
                                              top_retrieved_chunks=self._top_retrieved_chunks,
                                              dataset_processed_data_dir=self._dataset_processed_data_dir)

        logger.info("Reading instances")
        for d_id, doc_item in enumerate(doc_iterator):

            doc_id = doc_item.doc_id

            #fields_to_take = ["sentences_spans", "tokens", "pos", "lemmas", "ent", "ent_iob", "sentences_text", "entities", "coref_clusters"]
            fields_to_take_full = []
            fields_to_crop_with_span = ["tokens", "pos", "lemmas", "ent", "ent_iob"]

            curr_doc_questions = [x for x in questions_list if x["document_id"] == doc_id]
            for q_id, question_meta in enumerate(curr_doc_questions):
                question_with_ranking = doc_item.get_question_item(q_id)
                if question_with_ranking is not None:
                    assert question_meta["question"] == question_with_ranking["question"]

                context_sequence_parse = doc_item.get_context_parse(q_id,
                                                                    fields_to_take_full,
                                                                    fields_to_crop_with_span)
                extracted_entities = {}

                context_sequence = context_sequence_parse
                context_sequence["ent_with_iob"] = combine_parse_fields_if_both_exists_and_add_new_field(context_sequence,
                                                                                               ["ent_iob", "ent"],
                                                                                               [["O", ""], [""]],
                                                                                               "{0}_{1}",
                                                                                               default_result=[""] * len(context_sequence["tokens"]))

                context_sequence["tokens_with_ents"] = extract_and_map_entities(context_sequence,
                                                                               "tokens", "ent", "ent_iob",
                                                                               extracted_entities,
                                                                               default_result=context_sequence["tokens"],
                                                                               words_to_exclude_from_mapping=STOP_WORDS)

                # question sequence
                question_sequence = combine_sentences_parse(question_meta["question_parse"]["sentences"])
                question_sequence["ent_with_iob"] = combine_parse_fields_if_both_exists_and_add_new_field(question_sequence,
                                                                                                        ["ent_iob", "ent"],
                                                                                                        [["O", ""], [""]],
                                                                                                        "{0}_{1}",
                                                                                                        default_result=[""]
                                                                                                        * len(question_sequence["tokens"]))

                question_sequence["tokens_with_ents"] = extract_and_map_entities(question_sequence,
                                                                               "tokens", "ent", "ent_iob",
                                                                               extracted_entities,
                                                                               default_result=question_sequence["tokens"],
                                                                               words_to_exclude_from_mapping=STOP_WORDS)


                metadata = {
                    "document_id": question_meta["document_id"],
                    "question": question_meta["question"],
                    "answer1": question_meta["answer1"],
                    "answer2": question_meta["answer2"],
                    "target_references": [question_meta["answer1_tokenized"].split(),
                                          question_meta["answer2_tokenized"].split()],
                    # "extracted_entities": extracted_entities
                }

                # By default take answer1
                # We can also select the random answer1 or answer2 but this would disable reproducability.
                # In dev and test it will not be considered for the official evaluation
                # "target_references" from the metadata will be used
                target_sequence = combine_sentences_parse(question_meta["answer1_parse"]["sentences"])
                target_sequence["tokens_with_ents"] = extract_and_map_entities(target_sequence,
                                                                               "tokens", "ent", "ent_iob",
                                                                               extracted_entities,
                                                                               default_result=target_sequence["tokens"],
                                                                               words_to_exclude_from_mapping=STOP_WORDS)


                # target sequence 2
                target_sequence2 = combine_sentences_parse(question_meta["answer2_parse"]["sentences"])
                target_sequence2["tokens_with_ents"] = extract_and_map_entities(target_sequence2,
                                                                               "tokens", "ent", "ent_iob",
                                                                               extracted_entities,
                                                                               default_result=target_sequence2["tokens"],
                                                                               words_to_exclude_from_mapping=STOP_WORDS)

                # Use
                if isinstance(self._target_tokenizer._word_splitter, ReadSentenceParseTypeWise):
                    # set question to the input type
                    context_tokenized_text = [x.text for x in self._target_tokenizer.tokenize(context_sequence)]

                    metadata["context"] = " ".join(context_tokenized_text)

                    question_tokenized_text = [x.text for x in self._target_tokenizer.tokenize(question_sequence)]
                    metadata["question"] = " ".join(question_tokenized_text)

                    # set answer to the input type
                    answer1_tokenized_text = [x.text for x in self._target_tokenizer.tokenize(target_sequence)]
                    answer2_tokenized_text = [x.text for x in self._target_tokenizer.tokenize(target_sequence2)]
                    metadata["answer1"] = " ".join(answer1_tokenized_text)
                    metadata["answer2"] = " ".join(answer2_tokenized_text)
                    metadata["target_references"] = [answer1_tokenized_text,
                                                     answer2_tokenized_text]

                # this is a nasty check if it is a train dataset
                if file_path == "train":
                    target_sequences = [target_sequence, target_sequence2]
                    if self._handle_multiple_target_in_train == TARGET_HANDLE_ALL:
                        for target_sequence_curr in target_sequences:
                            yield self.text_to_instance(context_sequence, question_sequence, target_sequence_curr, metadata)
                    elif self._handle_multiple_target_in_train == TARGET_HANDLE_RANDOM:
                        # add answer 2 as an example as well - this doubles the train data!
                        target_sequence_curr = random.choice(target_sequences)
                        yield self.text_to_instance(context_sequence, question_sequence, target_sequence_curr, metadata)
                    elif self._handle_multiple_target_in_train == TARGET_HANDLE_ANSWER1:
                        yield self.text_to_instance(context_sequence, question_sequence, target_sequences[0], metadata)
                    elif self._handle_multiple_target_in_train == TARGET_HANDLE_ANSWER2:
                        yield self.text_to_instance(context_sequence, question_sequence, target_sequences[1], metadata)
                    else:
                        raise Exception("Something went wrong! You should not be here!")
                else:
                    # does not matter which target is passed! targets are evaluated from the metadata
                    # that contains all references
                    yield self.text_to_instance(context_sequence, question_sequence, target_sequence, metadata)


    @overrides
    def text_to_instance(self, context_parse: Dict[str, Any], question_parse: Dict[str, Any], target_parse: Dict[str, Any] = None,
                         metadata: Dict[str, Any]=None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields = {}

        # Create the instance fields
        if metadata is not None:
            fields["metadata"] = MetadataField(metadata)

        # context
        tokenized_context = self._context_tokenizer.tokenize(context_parse)
        if self._source_add_start_token:
            tokenized_context.insert(0, Token(START_SYMBOL))
        tokenized_context.append(Token(END_SYMBOL))

        if self._return_context_tokens_pointers:
            lowercase_tokens = False
            if "tokens" in self._context_token_indexers:
                lowercase_tokens = self._context_token_indexers["tokens"].lowercase_tokens

            context_tokens_text = [x.text for x in tokenized_context]
            _, unique_tokens_pointers, unique_tokens_list_lens = get_token_lookup_pointers(context_tokens_text,
                                                                                                            lowercase_tokens)

            context_tokens_pointers = ListField([ArrayField(np.asarray(x, dtype=np.int32), padding_value=-1) for x in unique_tokens_pointers])
            fields["context_tokens_pointers"] = context_tokens_pointers

        context_field = TextField(tokenized_context, self._context_token_indexers)
        fields["context_tokens"] = context_field

        # question
        tokenized_question = self._question_tokenizer.tokenize(question_parse)
        if self._source_add_start_token:
            tokenized_question.insert(0, Token(START_SYMBOL))
        tokenized_question.append(Token(END_SYMBOL))
        question_field = TextField(tokenized_question, self._question_token_indexers)
        fields["question_tokens"] = question_field

        if target_parse is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_parse)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)

            fields["target_tokens"] = target_field

        return Instance(fields)


