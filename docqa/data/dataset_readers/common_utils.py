"""
Common functions used by the data readers
"""
import json
from typing import Any, List, Dict

from allennlp.common import Params
from allennlp.data import Tokenizer, TokenIndexer


def tokenizer_dict_from_params(params: Params) -> 'Dict[str, Tokenizer]':  # type: ignore
    """
    ``Tokenizer`` can be used in a dictionary, with each ``Tokenizer`` getting a
    name.  The specification for this in a ``Params`` object is typically ``{"name" ->
    {tokenizer_params}}``.  This method reads that whole set of parameters and returns a
    dictionary suitable for use in a ``TextField``.

    Because default values for token indexers are typically handled in the calling class to
    this and are based on checking for ``None``, if there were no parameters specifying any
    tokenizers in the given ``params``, we return ``None`` instead of an empty dictionary.
    """
    tokenizers = {}
    for name, indexer_params in params.items():
        tokenizers[name] = Tokenizer.from_params(indexer_params)
    if tokenizers == {}:
        tokenizers = None
    return tokenizers


def token_indexer_dict_from_params(params: Params) -> 'Dict[str, TokenIndexer]':  # type: ignore
    """
    We typically use ``TokenIndexers`` in a dictionary, with each ``TokenIndexer`` getting a
    name.  The specification for this in a ``Params`` object is typically ``{"name" ->
    {indexer_params}}``.  This method reads that whole set of parameters and returns a
    dictionary suitable for use in a ``TextField``.

    Because default values for token indexers are typically handled in the calling class to
    this and are based on checking for ``None``, if there were no parameters specifying any
    token indexers in the given ``params``, we return ``None`` instead of an empty dictionary.
    """
    token_indexers = {}
    for name, indexer_params in params.items():
        token_indexers[name] = TokenIndexer.from_params(indexer_params)
    if token_indexers == {}:
        token_indexers = None
    return token_indexers


def get_key_and_value_by_key_match(key_value_map, key_to_try, default_key="any"):
    """
    This method looks for a key in a dictionary. If it is not found, an approximate key is selected by checking if the keys match with the end of the wanted key_to_try.
    The method is intended for use where the key is a relative file path!

    :param key_value_map:
    :param key_to_try:
    :param default_key:
    :return:
    """

    retrieved_value = None
    if key_to_try in key_value_map:
        retrieved_value = key_value_map[key_to_try]
        return retrieved_value

    if default_key is not None:
        retrieved_value = key_value_map.get(default_key, None)

    if len(key_value_map) == 1 and default_key in key_value_map:
        retrieved_value = key_value_map[default_key]
    else:
        for key in key_value_map.keys():
            key_clean = key.strip().strip("\"")
            key_clean = key_clean.replace("___dot___", ".")
            if key_clean == "any":
                continue
            if key_to_try.endswith(key_clean):
                retrieved_value = key_value_map[key]
                break

    if retrieved_value is None:
        raise ValueError(
            "key_value_map %s was not matched with a value! Even for the default key %s" % (key_to_try, default_key))

    return retrieved_value


def read_cn5_surface_text_from_json(input_file):
    """
    Reads conceptnet json and returns simple json only with text property that contains clean surfaceText.
    :param input_file: conceptnet json file
    :return: list of items with "text" key.
    """

    def clean_surface_text(surface_text):
        return surface_text.replace("[[", "").replace("]]", "")

    items = []
    for l_id, line in enumerate(open(input_file, mode="r")):
        item = json.loads(line.strip())
        text = clean_surface_text(item["surfaceText"])
        items.append({"text": text})

    return items


def read_json_flexible(input_file):
    """
    Reads json and returns simple json only with text property that contains clean text.
    Checks several different fields:
    - surfaceText  # conceptnet json
    - tkns
    :param input_file: conceptnet json file
    :return: list of items with "text" key.
    """

    def clean_surface_text(surface_text):
        return surface_text.replace("[[", "").replace("]]", "")

    items = []
    for l_id, line in enumerate(open(input_file, mode="r")):
        item = json.loads(line.strip())
        if "surfaceText" in item:  # conceptnet
            text = clean_surface_text(item["surfaceText"])
        elif "SCIENCE-FACT" in item:  # 1202HITS
            text = item["SCIENCE-FACT"]
        elif "Row Text" in item:  # WorldTree v1 - it is a typo but we want to handle these as well
            text = item["Row Text"]
        elif "Sentence" in item:  # Aristo Tuple KB v 5
            text = item["Sentence"].replace(".", "").replace("Some ", "").replace("Most ", "").replace("(part)", "")
        elif "fact_text" in item:
            text = item["fact_text"]
        else:
            raise ValueError(
                "Format is unknown. Does not contain of the fields: surfaceText, SCIENCE-FACT, Row Text, Sentence or Sentence!")

        items.append({"text": text})

    return items


def read_cn5_concat_subj_rel_obj_from_json(input_file):
    """
    Reads conceptnet json and returns simple json only with text property that contains clean surfaceText.
    :param input_file: conceptnet json file
    :return: list of items with "text" key.
    """

    def mask_rel(rel):
        return "@{0}@".format(rel.replace("/r/", "cn_"))

    items = []
    for l_id, line in enumerate(open(input_file, mode="r")):
        item = json.loads(line.strip())

        text = " ".join([item["surfaceStart"], mask_rel(item["rel"]), item["surfaceEnd"]])
        items.append({"text": text})

    return items


def load_json_from_file(file_name):
    """
    Loads items from a jsonl  file. Each line is expected to be a valid json.
    :param file_name: Jsonl file with single json object per line
    :return: List of serialized objects
    """
    items = []
    for line in open(file_name, mode="r"):
        item = json.loads(line.strip())
        items.append(item)

    return items


class KnowSourceManager():
    """Class that holds information about knowledge source"""

    def __init__(self,
                 knowledge_source_config_single=None,
                 know_reader=None,
                 know_rank_reader=None,
                 use_know_cache: bool = True,
                 max_facts_per_argument: Any = 0):
        self._knowledge_source_config_single = knowledge_source_config_single  # currently only one source is supported
        self._know_reader = know_reader
        self._know_rank_reader = know_rank_reader
        self._use_know_cache = use_know_cache
        self._max_facts_per_argument = max_facts_per_argument
        self._know_files_cache = {}
        self._know_rank_files_cache = {}

    def get_max_facts_per_argument(self, file_path):
        if isinstance(self._max_facts_per_argument, int):
            return self._max_facts_per_argument
        else:
            max_facts_to_take = get_key_and_value_by_key_match(self._max_facts_per_argument, file_path, "any")
            return max_facts_to_take


def any_in_set(values, zero_sts):
    for i, val in enumerate(values):
        if val in zero_sts[i]:
            return True
    return False

def combine_parse_fields_if_both_exists_and_add_new_field(parse,
                                                          field_names: List[str],
                                                          zero_values: List[List[str]],
                                                          new_field_format="{0}-{1}",
                                                          new_field_zero="",
                                                          default_result=None):
    """
    Given a parse, get the fields tokens and combine them to get a new field
    :param parse: Parse like {"tokens": ["Pesho visited Sofia"], "ner_type": ["PER", "", "LOC], "new_iob": ["B", "O", "B"]}
    :param field_names: List of input fields. Example ["ner_iob", "ner_type"]
    :param new_field_format: Format of the new string, given positional format ex. "{0}_{1}"
    :param default_result: Result if some of the fields does not exist
    :return: Result new tokens. e.g. ["B-PER", "", "B-LOC"]
    """
    # check if all input fields are available
    all_field_exist = True
    for fn in field_names:
        all_field_exist = all_field_exist and (fn in parse)

    if not all_field_exist:
        return default_result

    zeros = [set(x) for x in zero_values]
    # create the new_field
    new_field_tokens = [new_field_format.format(*x) if not any_in_set(x, zeros) else new_field_zero for x in zip(*[parse[fn] for fn in field_names])]

    return new_field_tokens


spacy_entities_mapping = {
"PERSON": "ENTITY_NAME",  #	People, including fictional.
"NORP": "NORP",  #	Nationalities or religious or political groups.
"FAC": "LOC",  #	Buildings, airports, highways, bridges, etc.
"ORG": "ENTITY_NAME",  #	Companies, agencies, institutions, etc.
"GPE": "LOC",  #	Countries, cities, states.
"LOC": "LOC",  #	Non-GPE locations, mountain ranges, bodies of water.
"PRODUCT": "ENTITY_NAME",  #	Objects, vehicles, foods, etc. (Not services.)
"EVENT": "EVENT",  #	Named hurricanes, battles, wars, sports events, etc.
"WORK_OF_ART": "ENTITY_NAME",  #	Titles of books, songs, etc.
"LAW": "LAW",  #	Named documents made into laws.
"LANGUAGE": "LANGUAGE",  #	Any named language.
"DATE": "DATE",  #	Absolute or relative dates or periods.
"TIME": "TIME",  #	Times smaller than a day.
"PERCENT": "PERCENT",  #	Percentage, including "%".
"MONEY": "MONEY",  #	Monetary values, including unit.
"QUANTITY": "QUANTITY",  #	Measurements, as of weight or distance.
"ORDINAL": "ORDINAL",  #	"first", "second", etc.
"CARDINAL": "CARDINAL",  #	Numerals that do not fall under another type.
}


def extract_and_map_entities(parse,
                              field_tokens,
                              field_type,
                              field_type_bio,
                              extracted_entiteis: Dict[str, Any],
                              field_type_zero_val="",
                              default_result=None,
                              use_type_mapping=True,
                              words_to_exclude_from_mapping=None):
    # check if all input fields are availeble
    new_values = []

    prev_entity_id = ""
    entity_id = ""

    if not "ENTITY_BIO_TO_ORIGINAL" in extracted_entiteis:
        extracted_entiteis["ENTITY_BIO_TO_ORIGINAL"] = {}

    for tkn_id, token in enumerate(parse[field_tokens]):
        val = token.lower()
        type_val = parse[field_type][tkn_id]
        if use_type_mapping:
            type_val = spacy_entities_mapping.get(type_val, type_val)

        val_with_type = "{0}_{1}".format(val, type_val)

        type_bio_val = parse[field_type_bio][tkn_id]

        if words_to_exclude_from_mapping is not None and val in words_to_exclude_from_mapping:
            entity_id = ""
            prev_entity_id = ""
        elif val_with_type in extracted_entiteis:
            entity_id = extracted_entiteis[val_with_type]["ID"]
            val = "{0}-{1}".format(type_bio_val, entity_id)
        elif type_bio_val == "I":
            entity_id = prev_entity_id
            val = "{0}-{1}".format(type_bio_val, entity_id)
        elif type_val != field_type_zero_val:
            type_cnt_key = "cnt_ent_type_{0}".format(type_val)
            type_cnt_val = extracted_entiteis.get(type_cnt_key, 0) + 1
            entity_id = "{0}{1}".format(type_val, type_cnt_val)
            extracted_entiteis[val_with_type] = {"ID": entity_id, "type": type_val}
            val = "{0}-{1}".format(type_bio_val, entity_id)
            extracted_entiteis[type_cnt_key] = type_cnt_val
        else:
            entity_id = ""
            prev_entity_id = ""

        prev_val = val
        prev_entity_id = entity_id
        prev_type_val = type_val
        prev_type_bio_val = type_bio_val

        new_values.append(val)

    return new_values


def combine_sentences_parse(sentences_parse):
    """
    Combines parsed sentences into a document parse.
    :param sentences_parse:
    :return: Document parse
    """
    doc_parsed = {}
    for sent_parse in sentences_parse:
        for key in sent_parse.keys():
            val = sent_parse[key]
            if not isinstance(val, list):
                continue

            if not key in doc_parsed:
                doc_parsed[key] = []

            doc_parsed[key].extend(sent_parse[key])

    return doc_parsed


def get_span_to_crop_passage_with_answer(passage_len, max_len, answer_start, answer_end):
    """
    Crop passsage in such a way that the answers are always in the cropped passahe
    :param passage_len: Original passage len
    :param max_len: Max len to be kept
    :param answer_start: The start of the original answer
    :param answer_end: The end of the original answer
    :return: Crop start, Crop end, new answer start, new answer end
    """

    if max_len is not None and max_len < passage_len:
        if max_len < answer_end:
            # We want to crop the passsage in such a way that the answer is always there!
            # We want to keep the location of the answer span to be proportional top the original passage
            span_end_percent = float(answer_end) / float(passage_len)
            answer_length = answer_end - answer_start

            new_passage_answer_end_from_end = int((1.0 - span_end_percent) * max_len)
            crop_end = min(passage_len, answer_end + new_passage_answer_end_from_end)
            crop_start = crop_end - max_len

            assert(crop_end - crop_start == max_len)
            new_answer_end = max_len - new_passage_answer_end_from_end
            new_answer_start = new_answer_end - answer_length
        else:
            crop_start = 0
            crop_end = max_len
            new_answer_end = answer_end
            new_answer_start = answer_start

        return crop_start, crop_end, new_answer_start, new_answer_end
    else:
        return None, None, None, None


def test_get_span_to_crop_passage_with_answer():
    orig_passage_tokens = "The balloon itself ultimately fails before the end, but makes it far enough across to get the protagonists to friendly lands, and eventually back to England, therefore succeeding".split()

    # TEST: The max len
    answer_start = 16
    answer_end = 19
    max_len = 20

    crop_start, crop_end, new_answer_start, new_answer_end = get_span_to_crop_passage_with_answer(len(orig_passage_tokens),
                                                                                                  max_len,
                                                                                                  answer_start,
                                                                                                  answer_end)

    cropped_passage = orig_passage_tokens[crop_start: crop_end]
    assert(len(cropped_passage) == max_len)

    original_answer = orig_passage_tokens[answer_start: answer_end]
    new_answer = cropped_passage[new_answer_start: new_answer_end]

    assert(" ".join(original_answer) == " ".join(new_answer))

