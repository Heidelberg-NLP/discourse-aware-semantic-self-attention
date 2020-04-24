from allennlp.data import Token
from overrides import overrides
from typing import List, Dict, Any

from allennlp.data.tokenizers.word_splitter import WordSplitter


def field_str_to_key_value_tuple(field, separator="->"):
    """
    This function converts a mapping like \"pos->pos_tags\" to (\"pos\", \"pos_tags\").
    This is used when mapping input field to a target (code name) in case that the input field
    does not map some expectation or convention. If we do not have mapping (ex. \"pos\", we assume (\"pos\", \"pos\").
    :param field: Field str to map
    :param separator: separator for the key and value.
    :return: Returns tuple like (\"pos\", \"pos_tags\") from \"pos->pos_tags\"
    """
    field_elems = [x.strip() for x in field.split("->")]

    mapping_tuple = None
    if len(field_elems) == 1:
        mapping_tuple = (field_elems[0], field_elems[0])
    elif len(field_elems) == 2:
        mapping_tuple = (field_elems[0], field_elems[1])
    else:
        raise ValueError("field should contain 1 or 2 fields. Example:"
                         "\"pos{0}pos_tags\" or just \"pos\""
                         "but we have field=\"{1}\" instead!".
                         format(separator, field))

    return mapping_tuple


@WordSplitter.register('read_sentence_parse_typewise')
class ReadSentenceParseTypeWise(WordSplitter):
    """
    A ``TokenWiseJsonParseSplitter`` reads the token information from the following json:
    {"tokens": ["Who", "commits", "suicide", "?"],
    "pos": ["WP", "VBZ", "NN", "."],
    "lemmas": ["who", "commit", "suicide", "?"],
    "ent": ["", "", "", ""],
    "ent_iob": ["O", "O", "O", "O"]}
    """
    def __init__(self,
                 fields: List[str],
                 ) -> None:

        self._fields_mapping = [field_str_to_key_value_tuple(fld) for fld in fields]

    # @overrides
    # def batch_split_words(self, sentences: List[str]) -> List[List[Token]]:
    #     return [_remove_spaces(tokens)
    #             for tokens in self.spacy.pipe(sentences, n_threads=-1)]

    @overrides
    def split_words(self, sentence: Dict[str, Any]) -> List[Token]:
        # We need spacy-like input for Token.
        fields_to_read = self._fields_mapping
        if "tokens_offsets" in sentence:
            fields_to_read = self._fields_mapping + [("tokens_offsets", "tokens_offsets")]

        fields, field_new = zip(*fields_to_read)

        input_tokens_attributes = zip(*[sentence.get(x, None) for x in fields])
        tokens_as_dict_mapped = [dict(zip(field_new, list(tkn_attrs))) for tkn_attrs in input_tokens_attributes]

        tokens = [
             Token(text=tkn.get("text", None),
              lemma=tkn.get("lemma", None),
              pos=tkn.get("pos", None),
              tag=tkn.get("tag", None),
              dep=tkn.get("dep", None),
              ent_type=tkn.get("ent_type", None),
              idx=tkn["tokens_offsets"][0] if "tokens_offsets" in tkn else None
              ) for tkn_id, tkn in enumerate(tokens_as_dict_mapped)]

        return tokens

if __name__ == "__main__":
    sentence_parse = {"tokens": ["Who", "commits", "suicide", "?"],
    "pos": ["WP", "VBZ", "NN", "."],
    "lemmas": ["who", "commit", "suicide", "?"],
    "ent": ["", "", "", ""],
    "ent_iob": ["O", "O", "O", "O"]}

    word_slitter = ReadSentenceParseTypeWise(["tokens->text", "pos->pos", "ent->ent_type"])

    tokens = word_slitter.split_words(sentence_parse)

    for i in range(4):
        assert tokens[i].text == sentence_parse["tokens"][i]
        assert tokens[i].pos_ == sentence_parse["pos"][i]
        assert tokens[i].ent_type_ == sentence_parse["ent"][i]
