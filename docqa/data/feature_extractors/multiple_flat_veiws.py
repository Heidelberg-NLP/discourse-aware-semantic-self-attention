from typing import List

import numpy as np
from overrides import overrides

from docqa.allennlp_custom.data import FeatureExtractor
from docqa.allennlp_custom.data.feature_extractors.feature_extractor import TokenWiseInteractionFeatureExtractor
from docqa.data.processing.text_semantic_graph import build_graph_with_srl


none_label = "@@NONE@@"

def trim_feats_list(feats_list, max_size):
    if len(feats_list) == 0:
        raise ValueError("feats_list length must be greater than 0")
    elif len(feats_list) >= max_size:
        return feats_list[:max_size]
    else:
        return feats_list


def pad_or_trim_feats_list(feats_list, max_size, pad_value=0):
    if len(feats_list) == 0:
        raise ValueError("feats_list length must be greater than 0")
    elif len(feats_list) >= max_size:
        return feats_list[:max_size]
    else:
        shape, dtype = feats_list[0].shape, feats_list[0].dtype

        pad_feats = [np.full(shape, pad_value, dtype=dtype) for i in range(max_size - len(feats_list))]
        return feats_list + pad_feats


@TokenWiseInteractionFeatureExtractor.register("multiple_flat_views")
class MultipleFlatViews(TokenWiseInteractionFeatureExtractor):
    def __init__(self,
                 feature_extractors: List[TokenWiseInteractionFeatureExtractor],
                 max_views: int,
                 labels_start_id: int = 1,  # this is for compatability with other runs
                 views_axis=0,
                 namespace="multiple",
                 ):
        super().__init__()

        self._max_views = max_views
        self._namespace = namespace
        self._views_axis = views_axis
        self._feature_extractors = feature_extractors
        self._vocab_feat_name2id = {}
        self._labels_start_id = labels_start_id

        # set vocab names
        curr_labels_start_id = self._labels_start_id
        for feature_extractor in feature_extractors:
            # update the starting label id for all feature extractors
            feature_extractor.set_vocab_feats_name2id_ids(curr_labels_start_id)
            curr_features_vocab = feature_extractor.get_vocab_feats_name2id()
            self._vocab_feat_name2id.update(curr_features_vocab)

            curr_labels_start_id += len(curr_features_vocab)



    @overrides
    def get_vocab_feats_name2id(self):
        return {k: v for k,v in self._vocab_feat_name2id.items()}

    @overrides
    def get_vocab_feats_id2name(self):
        return {v: k for k,v in self._vocab_feat_name2id.items()}

    @overrides
    def extract_features_raw(self, inputs):
        raise NotImplemented("inputs must be a parse containing `tokens` field!")


    @overrides
    def extract_features(self, inputs):
        src_feats = None
        sent_ids_mask = None

        for feature_extractor in self._feature_extractors:
            curr_src_feats, curr_sent_ids_mask = feature_extractor.extract_features(inputs)
            if src_feats is None:
                src_feats = curr_src_feats
                sent_ids_mask = curr_sent_ids_mask
            else:
                src_feats = np.concatenate([src_feats, curr_src_feats], self._views_axis)
                sent_ids_mask = np.concatenate([sent_ids_mask, curr_sent_ids_mask], self._views_axis)

        assert src_feats.shape == sent_ids_mask.shape, "Shapes of src_deats.shape={0} but sent_ids_mask.shape={1} " \
                                                       "should be the same but they are .".format(str(src_feats.shape),
                                                                                                  str(
                                                                                                      sent_ids_mask.shape))

        return src_feats, sent_ids_mask












