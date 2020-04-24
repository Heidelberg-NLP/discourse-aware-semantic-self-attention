from allennlp.common import Registrable
from overrides import overrides


class FeatureExtractor(Registrable):
    def __init__(self):
        pass

    def extract_features(self, inputs):
        raise NotImplementedError


class TokenWiseInteractionFeatureExtractor(Registrable):
    def __init__(self):
        super().__init__()
        pass

    def set_vocab_feats_name2id_ids(self, offset):
        raise NotImplementedError

    def get_vocab_feats_name2id(self):
        raise NotImplementedError

    def get_vocab_feats_id2name(self):
        raise NotImplementedError

    def extract_features(self, inputs):
        raise NotImplementedError

    def extract_features_raw(self, inputs):
        raise NotImplementedError

