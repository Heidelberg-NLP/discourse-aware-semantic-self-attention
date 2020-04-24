import numpy as np
from overrides import overrides

from docqa.allennlp_custom.data import FeatureExtractor
from docqa.allennlp_custom.data.feature_extractors.feature_extractor import TokenWiseInteractionFeatureExtractor
from docqa.data.feature_extractors.utils import trim_feats_list, pad_or_trim_feats_list
from docqa.data.processing.text_semantic_graph import build_graph_with_srl

def get_srl_inter_type(type, subtype):
    return "{0}__{1}".format(type.upper(), subtype.upper())

none_label = "@@NONE@@"

srl_types = ["SRL__B-V", "SRL__B-ARG0", "SRL__B-ARG1", "SRL__B-ARG2", "SRL__B-ARG3", "SRL__B-ARG4", "SRL__B-ARG5", "SRL__B-ARGA",
             "SRL__B-ARGM-ADJ", "SRL__B-ARGM-ADV", "SRL__B-ARGM-CAU", "SRL__B-ARGM-COM", "SRL__B-ARGM-DIR", "SRL__B-ARGM-DIS", "SRL__B-ARGM-DSP", "SRL__B-ARGM-EXT", "SRL__B-ARGM-GOL",
             "SRL__B-ARGM-LOC", "SRL__B-ARGM-LVB", "SRL__B-ARGM-MNR", "SRL__B-ARGM-MOD", "SRL__B-ARGM-NEG", "SRL__B-ARGM-PNC", "SRL__B-ARGM-PRD", "SRL__B-ARGM-PRP", "SRL__B-ARGM-PRR",
             "SRL__B-ARGM-PRX", "SRL__B-ARGM-REC", "SRL__B-ARGM-TMP",
             "SRL__B-C-ARG0", "SRL__B-C-ARG1", "SRL__B-C-ARG2", "SRL__B-C-ARG3", "SRL__B-C-ARG4", "SRL__B-C-ARGM-ADJ", "SRL__B-C-ARGM-ADV", "SRL__B-C-ARGM-CAU", "SRL__B-C-ARGM-COM",
             "SRL__B-C-ARGM-DIR", "SRL__B-C-ARGM-DIS", "SRL__B-C-ARGM-DSP", "SRL__B-C-ARGM-EXT", "SRL__B-C-ARGM-LOC", "SRL__B-C-ARGM-MNR", "SRL__B-C-ARGM-MOD",
             "SRL__B-C-ARGM-NEG", "SRL__B-C-ARGM-PRP", "SRL__B-C-ARGM-TMP",
             "SRL__B-R-ARG0", "SRL__B-R-ARG1", "SRL__B-R-ARG2", "SRL__B-R-ARG3", "SRL__B-R-ARG4", "SRL__B-R-ARG5", "SRL__B-R-ARGM-ADV", "SRL__B-R-ARGM-CAU", "SRL__B-R-ARGM-COM",
             "SRL__B-R-ARGM-DIR", "SRL__B-R-ARGM-EXT", "SRL__B-R-ARGM-GOL", "SRL__B-R-ARGM-LOC",
             "SRL__B-R-ARGM-MNR", "SRL__B-R-ARGM-MOD", "SRL__B-R-ARGM-PNC", "SRL__B-R-ARGM-PRD", "SRL__B-R-ARGM-PRP", "SRL__B-R-ARGM-TMP",
             "SRL__I-V", "SRL__I-ARG0", "SRL__I-ARG1", "SRL__I-ARG2", "SRL__I-ARG3", "SRL__I-ARG4", "SRL__I-ARG5", "SRL__I-ARGA",
             "SRL__I-ARGM-ADJ", "SRL__I-ARGM-ADV", "SRL__I-ARGM-CAU", "SRL__I-ARGM-COM", "SRL__I-ARGM-DIR", "SRL__I-ARGM-DIS", "SRL__I-ARGM-DSP", "SRL__I-ARGM-EXT", "SRL__I-ARGM-GOL",
             "SRL__I-ARGM-LOC", "SRL__I-ARGM-LVB", "SRL__I-ARGM-MNR", "SRL__I-ARGM-MOD", "SRL__I-ARGM-NEG", "SRL__I-ARGM-PNC", "SRL__I-ARGM-PRD", "SRL__I-ARGM-PRP", "SRL__I-ARGM-PRR",
             "SRL__I-ARGM-PRX", "SRL__I-ARGM-REC", "SRL__I-ARGM-TMP",
             "SRL__I-C-ARG0", "SRL__I-C-ARG1", "SRL__I-C-ARG2", "SRL__I-C-ARG3", "SRL__I-C-ARG4", "SRL__I-C-ARGM-ADJ", "SRL__I-C-ARGM-ADV", "SRL__I-C-ARGM-CAU", "SRL__I-C-ARGM-COM",
             "SRL__I-C-ARGM-DIR", "SRL__I-C-ARGM-DIS", "SRL__I-C-ARGM-DSP", "SRL__I-C-ARGM-EXT", "SRL__I-C-ARGM-LOC", "SRL__I-C-ARGM-MNR", "SRL__I-C-ARGM-MOD",
             "SRL__I-C-ARGM-NEG", "SRL__I-C-ARGM-PRP", "SRL__I-C-ARGM-TMP",
             "SRL__I-R-ARG0", "SRL__I-R-ARG1", "SRL__I-R-ARG2", "SRL__I-R-ARG3", "SRL__I-R-ARG4", "SRL__I-R-ARG5", "SRL__I-R-ARGM-ADV", "SRL__I-R-ARGM-CAU", "SRL__I-R-ARGM-COM",
             "SRL__I-R-ARGM-DIR", "SRL__I-R-ARGM-EXT", "SRL__I-R-ARGM-GOL", "SRL__I-R-ARGM-LOC",
             "SRL__I-R-ARGM-MNR", "SRL__I-R-ARGM-MOD", "SRL__I-R-ARGM-PNC", "SRL__I-R-ARGM-PRD", "SRL__I-R-ARGM-PRP", "SRL__I-R-ARGM-TMP"]


@TokenWiseInteractionFeatureExtractor.register("srl_flat_views")
class SRLFlatViews(TokenWiseInteractionFeatureExtractor):
    def __init__(self,
                 max_verbs: int,
                 labels_start_id: int = 1,  # this is for compatability with other runs
                 namespace: str = "srl",
                 views_axis: int = 0,
                 pad_views: bool = False,
                 use_mask: bool = True,
                 type: str = None
                 ):
        super().__init__()

        self._use_mask = use_mask
        self._max_verbs = max_verbs
        self._pad_views = pad_views
        self._namespace = namespace
        self._views_axis = views_axis
        self._labels_start_id = labels_start_id
        self._vocab_feat_name2id = {t: i + labels_start_id for i, t in enumerate(srl_types)}

    @overrides
    def set_vocab_feats_name2id_ids(self, offset):
        self._vocab_feat_name2id = {k: v - self._labels_start_id + offset for k, v in self._vocab_feat_name2id.items()}
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

        sent_ids_mask = []
        verb_feats = []
        
        all_tokens_cnt = 0

        # get number of tokens and sentence offsets
        for sent_id, sent in enumerate(inputs["sentences"]):
            all_tokens_cnt += len(sent["tokens"])
            sent_ids_mask.extend([sent_id + 1] * len(sent["tokens"]))

        sent_offset = 0
        for sent in inputs["sentences"]:
            curr_sent_offset = sent_offset
            sent_offset = sent_offset + len(sent["tokens"])
            if "srl" not in sent or "verbs" not in sent["srl"]:
                continue

            for verb_id, verb in enumerate(sent["srl"]["verbs"]):
                if len(verb_feats) <= verb_id:
                    verb_feats.append(np.zeros(all_tokens_cnt, dtype=np.int32))
                
                if "tags" in verb:
                    for tag_id, tag in enumerate(verb["tags"]):
                        verb_feats[verb_id][curr_sent_offset + tag_id] = self._vocab_feat_name2id.get("SRL__" + tag, 0)

        if len(verb_feats) == 0:
            verb_feats = [np.zeros(all_tokens_cnt, dtype=np.int32)]

        src_feats = [np.expand_dims(x, axis=self._views_axis) for x in pad_or_trim_feats_list(verb_feats,
                                                                                              self._max_verbs,
                                                                                              pad_to_max_size=self._pad_views)]

        if len(src_feats) > 1:
            sent_ids_mask = np.expand_dims(np.array(sent_ids_mask), axis=self._views_axis).repeat(len(src_feats),
                                                                                                  self._views_axis)
            src_feats = np.concatenate(src_feats, axis=self._views_axis)
        else:
            src_feats = src_feats[0]
            sent_ids_mask = np.array([sent_ids_mask])

        if not self._use_mask:
            sent_ids_mask = np.ones_like(src_feats)

        assert src_feats.shape == sent_ids_mask.shape, "Shapes of src_deats.shape={0} but sent_ids_mask.shape={1} " \
                                                       "should be the same but they are .".format(str(src_feats.shape),
                                                                                                  str(sent_ids_mask.shape))

        return src_feats, sent_ids_mask












