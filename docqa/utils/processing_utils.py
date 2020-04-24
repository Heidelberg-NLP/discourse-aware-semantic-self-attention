import json
import csv
import logging

from typing import List, Any



def clean_split(text, delim):
    """
    Split texts and removes unnecessary empty spaces or empty items.
    :param text: Text to split
    :param delim: Delimiter
    :return: List of splited strings
    """

    return [x1 for x1 in [x.strip() for x in text.split(delim)] if len(x1)>0]


def get_fields_from_txt(txt, field_delim=";", hier_delim="->", name_mapping_delim=":"):
    """
    Parses a field setup to dict of ("new_field_name", ["json_hier1", "json_heir2"]) given text
    :param txt: Example setting "new_field_name:json_hier1->json_heir2;new_field_name_2:json_hierA1->json_heirA2;"
    :param field_delim: Delimiter between field mappings. Default is semicolon ;
    :param hier_delim: Hierarchical delimiter between json fields. Default "->"
    :param name_mapping_delim: Delim between new field name and json fields. Default double dot ":"
    :return: Dictionary of (field_name, json_hier_fields_list)
    """

    splitted_fields = clean_split(txt, field_delim)

    named_splitted_fields = [(None, clean_split(x[0]), hier_delim) if len(x) == 1 else (x[0], clean_split(x[1], hier_delim)) for x in [clean_split(x, name_mapping_delim) for x in splitted_fields]]

    dict_name_to_key = {"__".join(k) if n is None else n :k for n,k in named_splitted_fields}

    return dict_name_to_key


def get_val_by_hier_key(json_item,
            hier_key:List[str],
            raise_key_error=False,
            default=None):
    """
    Gets a value of hierachical json fields. Does not support lists!
    :param json_item: Item to get the values from
    :param hier_key: List of hierarchical keys
    :param raise_key_error: Should it raise error on missing keys or return default field
    :param default: Default value if no key is found
    :return: Retrieved or Default value if no error is raised
    """
    curr_obj = json_item
    res_val = default

    found = True
    for fld in hier_key:
        if not fld in curr_obj:
            found = False
            if raise_key_error:
                raise KeyError("Key {0} not found in object json_item. {1}".format("->".join(["*%s*" % x if x==fld else x  for x in hier_key]),
                                                                                   "Starred item is where hierarchy lookup fails!" if len(hier_key) > 1 else "" ))
            break
        curr_obj = curr_obj[fld]

    if found:
        res_val = curr_obj

    return res_val



def try_set_val_by_hier_key(json_item,
                            hier_key: List[str],
                            value: Any,
                            create_hier_if_not_exists=True):
    """
    Gets a value of hierachical json fields. Does not support lists!
    :param json_item: Item to get the values from
    :param hier_key: List of hierarchical keys
    :param raise_key_error: Should it raise error on missing keys or return default field
    :param default: Default value if no key is found
    :return: Retrieved or Default value if no error is raised
    """
    curr_obj = json_item

    success = False
    for fid, fld in enumerate(hier_key):
        if isinstance(curr_obj, list):
            is_int = False
            try:
                fld_as_int = int(fld)
                is_int = True
            except ValueError:
                raise

            if is_int:
                if fid == (len(hier_key) - 1):  # if this is the last field in the hierarchy to set
                    curr_obj[fld_as_int] = value
                    success = True
                else:
                    curr_obj = curr_obj[fld_as_int]

        elif isinstance(curr_obj, dict):
            if not fld in curr_obj:
                if create_hier_if_not_exists:
                    curr_obj[fld] = {}
                else:
                    raise KeyError("Key {0} not found in object json_item. {1}".format(
                        "->".join(["*%s*" % x if x == fld else x for x in hier_key]),
                        "Starred *item* is where hierarchy lookup fails!" if len(hier_key) > 1 else ""))

            if fid == (len(hier_key) - 1):  # if this is the last field in the hierarchy to set
                curr_obj[fld] = value
                success = True
            else:
                curr_obj = curr_obj[fld]
        else:
            raise ValueError("Object {0} should be dictionary or list to do lookup.")


    return success


def test_try_set_val_by_hier_key():

    json = {}

    assert try_set_val_by_hier_key(json, ["value1", "value2", "value3"], value=[1, 2, 3], create_hier_if_not_exists=True)
    assert try_set_val_by_hier_key(json, ["value1", "value2", "value3", "2"], value=4)

    assert json["value1"]["value2"]["value3"][2] == 4

    failed = False
    try:
        try_set_val_by_hier_key(json, ["value1", "value2", "value3", "4"], value=4)
    except:
        failed = True
    assert failed


def get_fields_with_str_values_from_txt(txt, field_delim="|", hier_delim="->", value_mapping_delim="=>", ):
    """
    Parses a field setup to dict of (["json_hier1", "json_heir2"], value) given text
    :param txt: Example setting "json_hier1->json_heir2=>new_val;json_hierA1->json_heirA2=>new_val;"
    :param field_delim: Delimiter between field mappings. Default is semicolon ;
    :param hier_delim: Hierarchical delimiter between json fields. Default "->"
    :param value_mapping_delim: Delim between new field name and json fields. Default double dot ":"
    :return: Dictionary of (field_name, json_hier_fields_list)
    """

    transformations_list = clean_split(txt, field_delim)

    splitted_str_fields_with_values = [tuple(clean_split(field, value_mapping_delim)) for field in transformations_list]

    field_hier_with_str_values = [(clean_split(fv[0], hier_delim), fv[1]) for fv in splitted_str_fields_with_values]

    return field_hier_with_str_values


def assert_list_equal(l1, l2):
    return len(l1) == len(l2) and sorted(l1) == sorted(l2)


def test_get_fields_with_str_values_from_txt():
    # test multiple fields with multiple hierarchy
    transofrmations = "test1->test2->test3=>new_val1| test4->test5=>new_val2| test6=>  new_val3 |"

    expected = [(["test1", "test2", "test3"], "new_val1"),
                (["test4", "test5"], "new_val2"),
                (["test6"], "new_val3"),
                ]
    parsed = get_fields_with_str_values_from_txt(transofrmations, field_delim="|", hier_delim="->",
                                                 value_mapping_delim="=>")

    assert_list_equal(parsed[0][0], expected[0][0])
    assert parsed[0][1] == expected[0][1]

    assert_list_equal(parsed[1][0], expected[1][0])
    assert parsed[1][1] == expected[1][1]

    assert_list_equal(parsed[2][0], expected[2][0])
    assert parsed[2][1] == expected[2][1]


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
    # check if all input fields are availeble
    all_field_exist = True
    for fn in field_names:
        all_field_exist = all_field_exist and (fn in parse)

    if not all_field_exist:
        return default_result

    zeros = [set(x) for x in zero_values]
    # create the new_field
    new_field_tokens = [new_field_format.format(*x) if not any_in_set(x, zeros) else new_field_zero for x in zip(*[parse[fn] for fn in field_names])]

    return new_field_tokens


def load_json_list(input_file, filter_func=None):
    """
    Loads json list from a JSONL file
    :param input_file: Input file to load items from
    :param filter_func: If this is not None, the function should return
                        True if the item should be kept, False otherwise.
    :return: List ot dict objects
    """
    items = []
    with open(input_file, mode="r") as f_in:
        for line in f_in:
            item = json.loads(line.strip())
            if filter_func is not None:
                if filter_func(item):
                    items.append(item)
            else:
                items.append(item)

    return items




def read_csv_to_json_list(file_path, field_names, separator=",", json_filer_func=None):
    """
    Reads csv fiel to json list
    :param file_path: Path of the file
    :param field_names: Fields to read in the json
    :param separator: Separator for the csv. Default is comman (`,`).
    :return: List of dict objects with fields the loaded field_names
    """

    line_id = 0
    items_list = []
    with open(file_path, 'r') as data_file:
        logging.info("Reading custom instances from csv dataset at: %s", file_path)

        reader = csv.DictReader(data_file, field_names, delimiter=separator)
        skip_header = True
        for item_json in reader:
            if skip_header:
                skip_header = False
                continue
            line_id += 1

            if json_filer_func is not None:
                if json_filer_func(item_json):
                    items_list.append(item_json)

    return items_list


def batch_items(items, batch_size):
    """
    Batching sentences with a given batch_size
    :param items: List of items to batch
    :param batch_size: Batch size
    :return: Grouped sentences
    """
    sent_batches = []
    curr_batch = []
    for sentid, sent in enumerate(items):
        curr_batch.append(sent)
        if len(curr_batch) >= batch_size:
            sent_batches.append(curr_batch)
            curr_batch = []

    if len(curr_batch) > 0:
        sent_batches.append(curr_batch)

    return sent_batches


def get_val_by_hier_key(json_item,
            hier_key:List[str],
            raise_key_error=False,
            default=None):
    """
    Gets a value of hierachical json fields. Does not support lists!
    :param json_item: Item to get the values from
    :param hier_key: List of hierarchical keys
    :param raise_key_error: Should it raise error on missing keys or return default field
    :param default: Default value if no key is found
    :return: Retrieved or Default value if no error is raised
    """
    curr_obj = json_item
    res_val = default

    found = True
    for fld in hier_key:
        if not fld in curr_obj:
            found = False
            if raise_key_error:
                raise KeyError("Key {0} not found in object json_item. {1}".format("->".join(["*%s*" % x if x==fld else x  for x in hier_key]),
                                                                                   "Starred item is where hierarchy lookup fails!" if len(hier_key) > 1 else "" ))
            break
        curr_obj = curr_obj[fld]

    if found:
        res_val = curr_obj

    return res_val


def load_json_list(input_file, filter_func=None):
    """
    Loads json list from a JSONL file
    :param input_file: Input file to load items from
    :param filter_func: If this is not None, the function should return
                        True if the item should be kept, False otherwise.
    :return: List ot dict objects
    """
    items = []
    with open(input_file, mode="r") as f_in:
        for line in f_in:
            item = json.loads(line.strip())
            if filter_func is not None:
                if filter_func(item):
                    items.append(item)
            else:
                items.append(item)

    return items

def iterate_json_list(input_file, filter_func=None):
    """
    Iterates over json list from a JSONL file
    :param input_file: Input file to load items from
    :param filter_func: If this is not None, the function should return
                        True if the item should be kept, False otherwise.
    :return: nothing
    """
    with open(input_file, mode="r") as f_in:
        for line in f_in:
            item = json.loads(line.strip())
            if filter_func is not None:
                if filter_func(item):
                    yield item
            else:
                yield item


def check_if_parse_has_sentences_and_merge(parse_possibly_with_sentences):
    """
    Checks if the spacy parse has sentences and if it does, merges them.
    :param parse_possibly_with_sentences: Parse object
    :return: Parse with merged tokens
    """
    res = parse_possibly_with_sentences if "sentences" not in parse_possibly_with_sentences\
        else combine_sentences_parse(parse_possibly_with_sentences["sentences"])

    return res


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


def get_token_lookup_pointers(tokens_sequence, lowercase):
    """
    Given a list of tokens gather list of unique tokens with corresponding pointers to the occurence in the text
    and token occurence.
    Example:

    Input:
    tokens = ["A", "B", "C", "A", "B", "A"]
    lowercase = True

    Output:
    unique_tokens: ["a", "b", "c"]
    pointers: [ [0, 3, 5],
                [1, 4],
                [2] ]
    lens: [3, 2, 1]

    :param tokens_sequence: Token sequence
    :param lowercase: If we want to lowercase
    :return: List of unique tokens, List of pointers to the token occurence int he text, List of number of occurences
    """
    tokens_info_map = {}
    unique_tokens_list = []
    unique_tokens_pointers = []

    for id, token in enumerate(tokens_sequence):
        if lowercase:
            token = token.lower()

        token_info = tokens_info_map.get(token, None)
        if token_info is None:
            tkn_unique_id = len(unique_tokens_pointers)
            tkn_first_occurence = id
            tokens_info_map[token] = (tkn_unique_id, tkn_first_occurence)
            unique_tokens_pointers.append([tkn_first_occurence])
            unique_tokens_list.append(token)
        else:
            tkn_unique_id = token_info[0]
            unique_tokens_pointers[tkn_unique_id].append(id)

    unique_tokens_list_lens = [len(x) for x in unique_tokens_pointers]

    return unique_tokens_list, unique_tokens_pointers, unique_tokens_list_lens

