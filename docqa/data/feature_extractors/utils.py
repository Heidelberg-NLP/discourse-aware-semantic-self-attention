import numpy as np


def trim_feats_list(feats_list, max_size):
    if len(feats_list) == 0:
        raise ValueError("feats_list length must be greater than 0")
    elif len(feats_list) >= max_size:
        return feats_list[:max_size]
    else:
        return feats_list


def pad_or_trim_feats_list(feats_list, max_size, pad_to_max_size=True, pad_value=0):
    if len(feats_list) == 0:
        raise ValueError("feats_list length must be greater than 0")
    elif len(feats_list) >= max_size:
        return feats_list[:max_size]
    else:
        if pad_to_max_size:
            shape, dtype = feats_list[0].shape, feats_list[0].dtype

            pad_feats = [np.full(shape, pad_value, dtype=dtype) for i in range(max_size - len(feats_list))]
            return feats_list + pad_feats
        else:
            return feats_list
