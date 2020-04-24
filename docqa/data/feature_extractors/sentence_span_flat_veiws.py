from typing import List

import itertools

import numpy as np
from overrides import overrides

from docqa.allennlp_custom.data import FeatureExtractor
from docqa.allennlp_custom.data.feature_extractors.feature_extractor import TokenWiseInteractionFeatureExtractor
from docqa.data.feature_extractors.utils import trim_feats_list
from docqa.data.processing.text_semantic_graph import build_graph_with_srl

def get_srl_inter_type(type, subtype):
    return "{0}__{1}".format(type.upper(), subtype.upper())


def fill_span(feats, span, value):
    for token_id in range(span[0], span[1]):
        feats[token_id] = value


none_label = "@@NONE@@"

sentence_span_sense_labels = [
    "DR_NONEXP__Arg1",
    "DR_NONEXP__Arg2",
    "DR_NONEXP__Arg3",
    "DR_NONEXP__Arg4",
    "DR_NONEXP__Arg5",
    "DR_NONEXP__Arg6",
    "DR_NONEXP__Arg7",
    "DR_NONEXP__Arg8",
    "DR_NONEXP__Arg9",
    "DR_NONEXP__Arg10",
]


def get_sent_ids_mask(sentences):
    sent_ids_mask = []
    # get number of tokens and sentence offsets
    for sent_id, sent in enumerate(sentences):
        sent_ids_mask.extend([sent_id + 1] * len(sent["tokens"]))

    return sent_ids_mask


def generate_flat_feats_and_mask(inputs, span, feat_offset):
    all_tokens_cnt = 0

    # get number of tokens and sentence offsets
    for sent_id, sent in enumerate(inputs["sentences"]):
        all_tokens_cnt += len(sent["tokens"])

    feats = []
    mask = []
    for view_i in range(span):
        curr_view_feats = []
        mask_val = 1
        curr_view_mask = []
        for sent_id, sent in enumerate(inputs["sentences"]):
            tokens_cnt = len(sent["tokens"])
            sent_val = sent_id + view_i

            curr_sent_feat = sent_val % span + feat_offset
            sent_feats = [curr_sent_feat] * tokens_cnt
            curr_view_feats.extend(sent_feats)

            # generate mask by grouping consecutive sentences
            curr_view_mask.extend([mask_val] * tokens_cnt)
            if (curr_sent_feat - feat_offset) + 1 == span:
                mask_val += 1

        feats.append(curr_view_feats)
        mask.append(curr_view_mask)

    return feats, mask


@TokenWiseInteractionFeatureExtractor.register("sentence_span_flat_views")
class SentenceSpanFlatViews(TokenWiseInteractionFeatureExtractor):
    def __init__(self,
                 span: int,
                 max_views: int,
                 labels_start_id: int = 1,
                 namespace: str = "span_flat",
                 use_features_as_mask: bool = False,
                 use_mask: bool = True,
                 ):
        super().__init__()

        self._use_mask = use_mask
        self._use_features_as_mask = use_features_as_mask

        self._max_views = max_views
        self._namespace = namespace
        self._views_axis = 0
        self._span = span

        self._labels_start_id = labels_start_id
        self._vocab_feat_name2id = {t: i + labels_start_id for i, t in enumerate(sentence_span_sense_labels)}

    @overrides
    def set_vocab_feats_name2id_ids(self, offset):
        self._vocab_feat_name2id = {k: v - self._labels_start_id + offset for k, v in self._vocab_feat_name2id.items()}
        self._labels_start_id = offset

    @overrides
    def get_vocab_feats_name2id(self):
        return self._vocab_feat_name2id

    @overrides
    def get_vocab_feats_id2name(self):
        return {v: k for k, v in self._vocab_feat_name2id.items()}

    @overrides
    def extract_features_raw(self, inputs):
        raise NotImplemented("extract_features_raw is not supported")

    @overrides
    def extract_features(self, inputs):
        if not "sentences" in inputs:
            raise ValueError("inputs must be a parse containing `tokens` field!")

        all_tokens_cnt = 0

        # get number of tokens and sentence offsets
        for sent_id, sent in enumerate(inputs["sentences"]):
            all_tokens_cnt += len(sent["tokens"])

        src_feats, sent_mask = generate_flat_feats_and_mask(inputs, self._span, self._labels_start_id)
        src_feats = [np.expand_dims(np.asarray(x, dtype=np.int32), axis=self._views_axis) for x in src_feats]
        sent_mask = [np.expand_dims(np.asarray(x, dtype=np.int32), axis=self._views_axis) for x in sent_mask]

        src_feats = [x for x in trim_feats_list(src_feats, self._max_views)]
        if self._use_features_as_mask:
            # In this setting we use attention between same labels
            sent_mask = [x for x in trim_feats_list(src_feats, self._max_views)]
        else:
            sent_mask = [x for x in trim_feats_list(sent_mask, self._max_views)]

        if len(src_feats) > 1:
            src_feats = np.concatenate(src_feats, axis=self._views_axis)
            sent_mask = np.concatenate(sent_mask, axis=self._views_axis)
        else:
            src_feats = src_feats[0]
            sent_mask = sent_mask[0]

        if not self._use_mask:
            sent_mask = np.ones_like(src_feats)

        assert src_feats.shape == sent_mask.shape, "Shapes of src_deats.shape={0} but sent_ids_mask.shape={1} " \
                                                       "should be the same but they are .".format(str(src_feats.shape),
                                                                                                  str(sent_mask.shape))

        return src_feats, sent_mask


if __name__ == "__main__":
    parse = {"sentences":[
        {"tokens": ["This", "is", "sentnce", "1"]},
        {"tokens": ["There", "is", "another", "good", "sentence", "here"]},
        {"tokens": ["Sentence", "3", "is", "good", "."]},
        {"tokens": ["4th", "sentence", "is", "fine", "."]},
    ]}

    skip = 2
    mask = [1, 1, 1, 1,
            2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3,
            4, 4, 4, 4, 4]

    feats_1 = [1, 1, 1, 1,
               2, 2, 2, 2, 2, 2,
               1, 1, 1, 1, 1,
               2, 2, 2, 2, 2]

    feats_2 = [2, 2, 2, 2,
               1, 1, 1, 1, 1, 1,
               2, 2, 2, 2, 2,
               1, 1, 1, 1, 1]




    span = 2
    print("span={0}".format(span))
    feats, mask = generate_flat_feats_and_mask(parse, span=span, feat_offset=1)
    print(feats)
    print(mask)
    print("-" * 10)

    span = 3
    print("span={0}".format(span))
    feats, mask = generate_flat_feats_and_mask(parse, span=span, feat_offset=1)
    print(feats)
    print(mask)
    print("-" * 10)

    span = 4
    print("span={0}".format(span))
    feats, mask = generate_flat_feats_and_mask(parse, span=span, feat_offset=1)
    print(feats)
    print(mask)
    print("-" * 10)

