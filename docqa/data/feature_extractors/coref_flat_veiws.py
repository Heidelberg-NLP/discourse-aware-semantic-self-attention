import numpy as np
from overrides import overrides

from docqa.allennlp_custom.data import FeatureExtractor
from docqa.allennlp_custom.data.feature_extractors.feature_extractor import TokenWiseInteractionFeatureExtractor
from docqa.data.feature_extractors.utils import trim_feats_list
from docqa.data.processing.text_semantic_graph import build_graph_with_srl

def get_srl_inter_type(type, subtype):
    return "{0}__{1}".format(type.upper(), subtype.upper())

none_label = "@@NONE@@"

@TokenWiseInteractionFeatureExtractor.register("coref_flat_views")
class CorefFlatViews(TokenWiseInteractionFeatureExtractor):
    def __init__(self,
                 max_views: int,
                 namespace: str = "coref",
                 pad_views: bool = False,
                 views_axis=0,
                 ):
        super().__init__()

        self._max_views = max_views
        self._namespace = namespace
        self._views_axis = views_axis
        self._pad_views = pad_views
        self._vocab_feat_name2id = {}

    @overrides
    def set_vocab_feats_name2id_ids(self, offset):
        self._vocab_feat_name2id = {}
        self._labels_start_id = offset

    @overrides
    def get_vocab_feats_name2id(self):
        return self._vocab_feat_name2id

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

        all_tokens_cnt = 0

        # get number of tokens and sentence offsets
        for sent_id, sent in enumerate(inputs["sentences"]):
            all_tokens_cnt += len(sent["tokens"])

        coref_feats = [np.zeros(all_tokens_cnt, dtype=np.int32)]
        sent_ids_mask = np.zeros(all_tokens_cnt, dtype=np.int32)

        if "coref_clusters" in inputs:
            for coref_id, coref_cluster in enumerate(inputs["coref_clusters"]):
                cluster_label = coref_id + 1
                if "mentions" not in coref_cluster:
                    continue

                for mention in coref_cluster["mentions"]:
                    for tkn_id in range(mention["start"], mention["end"]):
                        sent_ids_mask[tkn_id] = cluster_label

        if len(coref_feats) == 0:
            coref_feats = [np.zeros(all_tokens_cnt, dtype=np.int32)]

        src_feats = [np.expand_dims(x, axis=self._views_axis) for x in trim_feats_list(coref_feats, self._max_views)]

        if len(src_feats) > 1:
            sent_ids_mask = np.expand_dims(np.array(sent_ids_mask), axis=self._views_axis).repeat(len(src_feats), self._views_axis)
            src_feats = np.concatenate(src_feats, axis=self._views_axis)
        else:
            src_feats = src_feats[0]
            sent_ids_mask = np.array([sent_ids_mask])

        if self._pad_views and self._max_views < src_feats.shape[1]:
            src_feats = src_feats.repeat(self._max_views, self._views_axis)
            sent_ids_mask = sent_ids_mask.repeat(self._max_views, self._views_axis)

        assert src_feats.shape == sent_ids_mask.shape, "Shapes of src_feats.shape={0} but sent_ids_mask.shape={1} " \
                                                       "should be the same but they are .".format(str(src_feats.shape),
                                                                                                  str(sent_ids_mask.shape))

        return src_feats, sent_ids_mask












