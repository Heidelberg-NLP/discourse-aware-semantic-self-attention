import numpy as np
from overrides import overrides

from docqa.allennlp_custom.data import FeatureExtractor
from docqa.allennlp_custom.data.feature_extractors.feature_extractor import TokenWiseInteractionFeatureExtractor
from docqa.data.processing.text_semantic_graph import build_graph_with_srl

def get_srl_inter_type(type, subtype):
    return "{0}__{1}".format(type.upper(), subtype.upper())


def set_array_square(a, left_top_corner, right_bottom_corner, value):
    for row in range(left_top_corner[0], right_bottom_corner[0] + 1):
        for col in range(left_top_corner[1], right_bottom_corner[1] + 1):
            a[row, col] = value

def convert_srl_graph_to_token_interactions(context_size, srl_graph, srl_interaction_vocab):
    verb_views = {}
    for edge in srl_graph["edges"]:
        verb_id = edge.get("verb_id", None)
        if verb_id is None:
            continue

        if verb_id not in verb_views:
            verb_views[verb_id] = {"src": np.zeros((context_size, context_size), dtype=np.int32),
                                  "tgt": np.zeros((context_size, context_size), dtype=np.int32)}

        source_node = srl_graph["nodes"][edge["source"]]
        target_node = srl_graph["nodes"][edge["target"]]

        #relation = srl_interaction_vocab.get(edge["rel"], 0)
        src_rel = srl_interaction_vocab.get(get_srl_inter_type(source_node["type"], source_node["sub_type"]), 0)
        tgt_rel = srl_interaction_vocab.get(get_srl_inter_type(target_node["type"], target_node["sub_type"]), 0)

        left_top = (source_node["span_start"], target_node["span_start"] - 1)
        right_bottom = (source_node["span_end"], target_node["span_end"] - 1)
        set_array_square(verb_views[verb_id]["src"], left_top, right_bottom, src_rel)
        set_array_square(verb_views[verb_id]["tgt"], left_top, right_bottom, tgt_rel)

    return verb_views


def get_srl_interactions(parse_json, srl_interaction_vocab):
    graph = build_graph_with_srl(parse_json,
                                 add_rel_between_args=True,
                                 include_prev_verb_rel=False)

    context_size = sum([len(sent["tokens"]) for sent in parse_json["sentences"]])
    tokenwise_labels = convert_srl_graph_to_token_interactions(context_size, graph, srl_interaction_vocab)

    del graph

    return tokenwise_labels

none_label = "@@NONE@@"
#max_attention_heads = 30
srl_types = ["SRL__V", "SRL__ARG0", "SRL__ARG1", "SRL__ARG2", "SRL__ARG3", "SRL__ARG4", "SRL__ARG5", "SRL__ARGA",
             "SRL__ARGM-ADJ", "SRL__ARGM-ADV", "SRL__ARGM-CAU", "SRL__ARGM-COM", "SRL__ARGM-DIR", "SRL__ARGM-DIS", "SRL__ARGM-DSP", "SRL__ARGM-EXT", "SRL__ARGM-GOL",
             "SRL__ARGM-LOC", "SRL__ARGM-LVB", "SRL__ARGM-MNR", "SRL__ARGM-MOD", "SRL__ARGM-NEG", "SRL__ARGM-PNC", "SRL__ARGM-PRD", "SRL__ARGM-PRP", "SRL__ARGM-PRR",
             "SRL__ARGM-PRX", "SRL__ARGM-REC", "SRL__ARGM-TMP",
             "SRL__C-ARG0", "SRL__C-ARG1", "SRL__C-ARG2", "SRL__C-ARG3", "SRL__C-ARG4", "SRL__C-ARGM-ADJ", "SRL__C-ARGM-ADV", "SRL__C-ARGM-CAU", "SRL__C-ARGM-COM",
             "SRL__C-ARGM-DIR", "SRL__C-ARGM-DIS", "SRL__C-ARGM-DSP", "SRL__C-ARGM-EXT", "SRL__C-ARGM-LOC", "SRL__C-ARGM-MNR", "SRL__C-ARGM-MOD",
             "SRL__C-ARGM-NEG", "SRL__C-ARGM-PRP", "SRL__C-ARGM-TMP",
             "SRL__R-ARG0", "SRL__R-ARG1", "SRL__R-ARG2", "SRL__R-ARG3", "SRL__R-ARG4", "SRL__R-ARG5", "SRL__R-ARGM-ADV", "SRL__R-ARGM-CAU", "SRL__R-ARGM-COM",
             "SRL__R-ARGM-DIR", "SRL__R-ARGM-EXT", "SRL__R-ARGM-GOL", "SRL__R-ARGM-LOC",
             "SRL__R-ARGM-MNR", "SRL__R-ARGM-MOD", "SRL__R-ARGM-PNC", "SRL__R-ARGM-PRD", "SRL__R-ARGM-PRP", "SRL__R-ARGM-TMP"]


def pad_or_trim_feats_list(feats_list, max_size, pad_value=0):
    if len(feats_list) == 0:
        raise ValueError("feats_list length must be greater than 0")
    elif len(feats_list) >= max_size:
        return feats_list[:max_size]
    else:
        shape, dtype = feats_list[0].shape, feats_list[0].dtype

        pad_feats = [np.full(shape, pad_value, dtype=dtype) for i in range(max_size - len(feats_list))]
        return feats_list + pad_feats


@FeatureExtractor.register("srl_interaction_views")
class SRLInteractionViews(TokenWiseInteractionFeatureExtractor):
    def __init__(self,
                 type: str,
                 max_verbs: int,
                 namespace: str = "srl",
                 views_axis=0,
                 ):
        super().__init__()

        self._max_verbs = max_verbs
        self._namespace = namespace
        self._views_axis = views_axis
        self._vocab_feat_name2id = {t: i for i, t in enumerate(
            [none_label]  # + ["Att{0:03d}".format(x) for x in range(max_attention_heads)]\
            + srl_types)}


    @overrides
    def get_vocab_feats_name2id(self):
        return {v: k for k,v in self._vocab_feat_name2id.items()}

    @overrides
    def get_vocab_feats_id2name(self):
        return {v: k for k,v in self._vocab_feat_name2id.items()}

    @overrides
    def extract_features_raw(self, inputs):
        if not "tokens" in inputs:
            raise ValueError("inputs must be a parse containing `tokens` field!")

        graph = build_graph_with_srl(inputs,
                                     add_rel_between_args=False,
                                     include_prev_verb_rel=False)

        return graph

    @overrides
    def extract_features(self, inputs):
        if not "sentences" in inputs:
            raise ValueError("inputs must be a parse containing `tokens` field!")

        feats = get_srl_interactions(inputs, self._vocab_feat_name2id)
        verb_feats = [x[1] for x in sorted([(v_id, v_feats) for v_id, v_feats in feats.items()], key=lambda a: a[0])]
        if len(verb_feats) == 0:
            tokens_cnt = sum([len(x["tokens"]) for x in inputs["sentences"]])

            verb_feats = [{"src": np.full((tokens_cnt, tokens_cnt), 0, dtype=np.int32),
                           "tgt": np.full((tokens_cnt, tokens_cnt), 0, dtype=np.int32)
                           }]

        src_feats = np.copy(np.concatenate(np.expand_dims(pad_or_trim_feats_list([x["src"] for x in verb_feats], self._max_verbs), axis=self._views_axis), axis=self._views_axis))
        tgt_feats = np.copy(np.concatenate(np.expand_dims(pad_or_trim_feats_list([x["tgt"] for x in verb_feats], self._max_verbs), axis=self._views_axis), axis=self._views_axis))

        del feats

        return src_feats, tgt_feats












