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

RELATION_TYPES = ['Explicit', 'Implicit', 'AltLex', 'EntRel', 'NoRel']
en_discr_rel_senses_distint = [
    'Temporal.Asynchronous.Precedence',
    'Temporal.Asynchronous.Succession',
    'Temporal.Synchrony',
    'Contingency.Cause.Reason',
    'Contingency.Cause.Result',
    'Contingency.Condition',
    'Comparison.Contrast',
    'Comparison.Concession',
    'Expansion.Conjunction',
    'Expansion.Instantiation',
    'Expansion.Restatement',
    'Expansion.Alternative',
    'Expansion.Alternative.Chosen alternative',
    'Expansion.Exception',
    'EntRel',
    ]

disc_rel_sense_labels = [
    "DR_EXP__Arg1", "DR_EXP__Arg2", "DR_EXP__Conn",
    "DR_NONEXP__Arg1", "DR_NONEXP__Arg2", "DR_NONEXP__Conn",
    "DR_EXP__Comparison.Concession__Arg1", "DR_EXP__Comparison.Concession__Arg2", "DR_EXP__Comparison.Concession__Conn",
    "DR_EXP__Comparison.Contrast__Arg1", "DR_EXP__Comparison.Contrast__Arg2", "DR_EXP__Comparison.Contrast__Conn",
    "DR_EXP__Contingency.Cause.Reason__Arg1", "DR_EXP__Contingency.Cause.Reason__Arg2", "DR_EXP__Contingency.Cause.Reason__Conn",
    "DR_EXP__Contingency.Cause.Result__Arg1", "DR_EXP__Contingency.Cause.Result__Arg2", "DR_EXP__Contingency.Cause.Result__Conn",
    "DR_EXP__Contingency.Condition__Arg1", "DR_EXP__Contingency.Condition__Arg2", "DR_EXP__Contingency.Condition__Conn",
    "DR_EXP__EntRel__Arg1", "DR_EXP__EntRel__Arg2", "DR_EXP__EntRel__Conn",
    "DR_EXP__Expansion.Alternative.Chosen alternative__Arg1", "DR_EXP__Expansion.Alternative.Chosen alternative__Arg2", "DR_EXP__Expansion.Alternative.Chosen alternative__Conn",
    "DR_EXP__Expansion.Alternative__Arg1", "DR_EXP__Expansion.Alternative__Arg2", "DR_EXP__Expansion.Alternative__Conn",
    "DR_EXP__Expansion.Conjunction__Arg1", "DR_EXP__Expansion.Conjunction__Arg2", "DR_EXP__Expansion.Conjunction__Conn",
    "DR_EXP__Expansion.Exception__Arg1", "DR_EXP__Expansion.Exception__Arg2", "DR_EXP__Expansion.Exception__Conn",
    "DR_EXP__Expansion.Instantiation__Arg1", "DR_EXP__Expansion.Instantiation__Arg2", "DR_EXP__Expansion.Instantiation__Conn",
    "DR_EXP__Expansion.Restatement__Arg1", "DR_EXP__Expansion.Restatement__Arg2", "DR_EXP__Expansion.Restatement__Conn",
    "DR_EXP__Temporal.Asynchronous.Precedence__Arg1", "DR_EXP__Temporal.Asynchronous.Precedence__Arg2", "DR_EXP__Temporal.Asynchronous.Precedence__Conn",
    "DR_EXP__Temporal.Asynchronous.Succession__Arg1", "DR_EXP__Temporal.Asynchronous.Succession__Arg2", "DR_EXP__Temporal.Asynchronous.Succession__Conn",
    "DR_EXP__Temporal.Synchrony__Arg1", "DR_EXP__Temporal.Synchrony__Arg2", "DR_EXP__Temporal.Synchrony__Conn",
    "DR_NONEXP__Comparison.Concession__Arg1", "DR_NONEXP__Comparison.Concession__Arg2", "DR_NONEXP__Comparison.Concession__Conn",
    "DR_NONEXP__Comparison.Contrast__Arg1", "DR_NONEXP__Comparison.Contrast__Arg2", "DR_NONEXP__Comparison.Contrast__Conn",
    "DR_NONEXP__Contingency.Cause.Reason__Arg1", "DR_NONEXP__Contingency.Cause.Reason__Arg2", "DR_NONEXP__Contingency.Cause.Reason__Conn",
    "DR_NONEXP__Contingency.Cause.Result__Arg1", "DR_NONEXP__Contingency.Cause.Result__Arg2", "DR_NONEXP__Contingency.Cause.Result__Conn",
    "DR_NONEXP__Contingency.Condition__Arg1", "DR_NONEXP__Contingency.Condition__Arg2", "DR_NONEXP__Contingency.Condition__Conn",
    "DR_NONEXP__EntRel__Arg1", "DR_NONEXP__EntRel__Arg2", "DR_NONEXP__EntRel__Conn",
    "DR_NONEXP__Expansion.Alternative.Chosen alternative__Arg1", "DR_NONEXP__Expansion.Alternative.Chosen alternative__Arg2", "DR_NONEXP__Expansion.Alternative.Chosen alternative__Conn",
    "DR_NONEXP__Expansion.Alternative__Arg1", "DR_NONEXP__Expansion.Alternative__Arg2", "DR_NONEXP__Expansion.Alternative__Conn",
    "DR_NONEXP__Expansion.Conjunction__Arg1", "DR_NONEXP__Expansion.Conjunction__Arg2", "DR_NONEXP__Expansion.Conjunction__Conn",
    "DR_NONEXP__Expansion.Exception__Arg1", "DR_NONEXP__Expansion.Exception__Arg2", "DR_NONEXP__Expansion.Exception__Conn",
    "DR_NONEXP__Expansion.Instantiation__Arg1", "DR_NONEXP__Expansion.Instantiation__Arg2", "DR_NONEXP__Expansion.Instantiation__Conn",
    "DR_NONEXP__Expansion.Restatement__Arg1", "DR_NONEXP__Expansion.Restatement__Arg2", "DR_NONEXP__Expansion.Restatement__Conn",
    "DR_NONEXP__Temporal.Asynchronous.Precedence__Arg1", "DR_NONEXP__Temporal.Asynchronous.Precedence__Arg2", "DR_NONEXP__Temporal.Asynchronous.Precedence__Conn",
    "DR_NONEXP__Temporal.Asynchronous.Succession__Arg1", "DR_NONEXP__Temporal.Asynchronous.Succession__Arg2", "DR_NONEXP__Temporal.Asynchronous.Succession__Conn",
    "DR_NONEXP__Temporal.Synchrony__Arg1", "DR_NONEXP__Temporal.Synchrony__Arg2", "DR_NONEXP__Temporal.Synchrony__Conn"
]

def get_sent_ids_mask(sentences):
    sent_ids_mask = []
    # get number of tokens and sentence offsets
    for sent_id, sent in enumerate(sentences):
        sent_ids_mask.extend([sent_id + 1] * len(sent["tokens"]))

    return sent_ids_mask


@TokenWiseInteractionFeatureExtractor.register("sdp_flat_views")
class SDPFlatViews(TokenWiseInteractionFeatureExtractor):
    def __init__(self,
                 use_nonexplicit: bool,
                 max_nonexplicit_views: int,
                 max_views: int,
                 labels_start_id: int = 1,
                 use_explicit: bool = False,
                 max_explicit_views: int = 0,
                 use_senses_for_tags: bool = True,
                 use_features_as_mask: bool = False,
                 namespace: str = "sdp",
                 use_mask: bool = True,
                 ):
        super().__init__()

        self._use_mask = use_mask
        self._use_senses_for_tags = use_senses_for_tags
        self._use_features_as_mask = use_features_as_mask
        self._use_explicit = use_explicit
        # if use_explicit:
        #     raise ValueError("`use_explicit` relations are not currently supported!")

        self._max_explicit_views = 0 if not use_explicit else max_explicit_views

        self._use_nonexplicit = use_nonexplicit
        self._max_nonexplicit_views = 0 if not use_nonexplicit else max_nonexplicit_views

        max_views_expected = self._max_explicit_views + self._max_nonexplicit_views
        if self._max_explicit_views + self._max_nonexplicit_views != max_views:
            raise ValueError("`max_views` ({0}) should be equal to `max_explicit_views` ({1}, "
                             "because `use_explicit` is {2}) "
                             "+ `max_nonexplicit_views` ({3}, because `use_nonexplicit` is {4}) = {5}. "
                             "This assertions goal is to make sure you know tht total `max_views`!"
                             .format(max_views, self._max_explicit_views, self._use_explicit,
                                     self._max_nonexplicit_views, self._use_nonexplicit, max_views_expected))

        self._max_views = max_views

        self._namespace = namespace
        self._views_axis = 0

        self._labels_start_id = labels_start_id
        self._vocab_feat_name2id = {t: i + labels_start_id for i, t in enumerate(disc_rel_sense_labels)}

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

        feats = []
        sent_ids_mask = []

        dr_type_to_label = {"Implicit": "NONEXP", "Explicit": "EXP"}
        if "sdp" in inputs:
            if self._use_nonexplicit:
                # odd
                feats.append(np.zeros(all_tokens_cnt, dtype=np.int32))
                sent_ids_mask.append(np.zeros(all_tokens_cnt, dtype=np.int32))

                #even
                feats.append(np.zeros(all_tokens_cnt, dtype=np.int32))
                sent_ids_mask.append(np.zeros(all_tokens_cnt, dtype=np.int32))

                for anno_id, annotation in enumerate(inputs["sdp"]):
                    if annotation["Type"] != "Implicit":
                        continue

                    arg1_sent_id = annotation["Arg1"]["Sent"]
                    feat_id = (arg1_sent_id + 1) % 2
                    mask_value = arg1_sent_id + 1

                    for arg_name in ["Arg1", "Arg2", "Conn"]:
                        span = annotation[arg_name]["Span"]
                        if len(span) == 0:
                            continue

                        dr_sense_label = "__" + annotation["Sense"] if self._use_senses_for_tags else ""
                        label = "DR_{0}{1}__{2}".format(dr_type_to_label[annotation["Type"]],
                                                       dr_sense_label,
                                                       arg_name)
                        label_id = self._vocab_feat_name2id.get(label, 0)
                        fill_span(feats[feat_id], span, label_id)
                        fill_span(sent_ids_mask[feat_id], span, mask_value)

            if self._use_explicit:
                mask_by_sent_ids = get_sent_ids_mask(inputs["sentences"])

                actual_exp_view_id = []  # here we maintain the mapping to the views
                for exp_v in range(self._max_explicit_views):
                    actual_exp_view_id.append(len(feats))
                    feats.append(np.zeros(all_tokens_cnt, dtype=np.int32))
                    sent_ids_mask.append(np.asarray(mask_by_sent_ids, dtype=np.int32))

                sentid_to_viewid = {}
                for anno_id, annotation in enumerate(inputs["sdp"]):
                    if annotation["Type"] != "Explicit":
                        continue

                    # For each sentences we have maximum number of explicit views (maximum number of connectives).
                    # We have to count the used views and stop adding features for the connectives > _max_explicit_views
                    arg1_sent_id = annotation["Arg1"]["Sent"]
                    view_id = sentid_to_viewid.get(arg1_sent_id, -1)
                    view_id += 1
                    sentid_to_viewid[arg1_sent_id] = view_id
                    if view_id > (self._max_explicit_views -1):
                        continue

                    feat_id = actual_exp_view_id[view_id]
                    mask_value = arg1_sent_id + 1

                    for arg_name in ["Arg1", "Arg2", "Conn"]:
                        span = annotation[arg_name]["Span"]
                        if len(span) == 0:
                            continue

                        dr_sense_label = "__" + annotation["Sense"] if self._use_senses_for_tags else ""
                        label = "DR_{0}{1}__{2}".format(dr_type_to_label[annotation["Type"]],
                                                        dr_sense_label,
                                                        arg_name)
                        label_id = self._vocab_feat_name2id.get(label, 0)
                        fill_span(feats[feat_id], span, label_id)
                        fill_span(sent_ids_mask[feat_id], span, mask_value)

        if len(feats) == 0:
            feats = [np.zeros(all_tokens_cnt, dtype=np.int32)]

        if len(sent_ids_mask) == 0:
            sent_ids_mask = [np.ones(all_tokens_cnt, dtype=np.int32)]

        src_feats = [np.expand_dims(x, axis=self._views_axis) for x in trim_feats_list(feats, self._max_views)]
        if self._use_features_as_mask:
            # In this setting we use attention between same labels
            sent_ids_mask = [np.expand_dims(x, axis=self._views_axis) for x in trim_feats_list(feats, self._max_views)]
        else:
            sent_ids_mask = [np.expand_dims(x, axis=self._views_axis) for x in trim_feats_list(sent_ids_mask, self._max_views)]

        if len(src_feats) > 1:
            src_feats = np.concatenate(src_feats, axis=self._views_axis)
            sent_ids_mask = np.concatenate(sent_ids_mask, axis=self._views_axis)
        else:
            src_feats = src_feats[0]
            sent_ids_mask = sent_ids_mask[0]

        if not self._use_mask:
            sent_ids_mask = np.ones_like(src_feats)

        assert src_feats.shape == sent_ids_mask.shape, "Shapes of src_deats.shape={0} but sent_ids_mask.shape={1} " \
                                                       "should be the same but they are .".format(str(src_feats.shape),
                                                                                                  str(sent_ids_mask.shape))

        return src_feats, sent_ids_mask












