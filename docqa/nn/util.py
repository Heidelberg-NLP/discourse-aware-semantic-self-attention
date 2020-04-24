import numpy as np
import torch
from allennlp.nn.util import get_final_encoder_states

def to_cuda(input: torch.Tensor, move_to_cuda):
    if move_to_cuda and not input.is_cuda:
        return input.cuda()
    else:
        return input

def float_fill(shape, val, use_cuda):
    if use_cuda:
        return torch.cuda.FloatTensor(*shape).fill_(val)
    else:
        return torch.FloatTensor(*shape).fill_(val)

def long_fill(shape, val, use_cuda):
    if use_cuda:
        return torch.cuda.LongTensor(*shape).fill_(val)
    else:
        return torch.LongTensor(*shape).fill_(val)


def seq2vec_seq_aggregate(seq_tensor, mask, aggregate, bidirectional, dim=1):
    """
        Takes the aggregation of sequence tensor

        :param seq_tensor: Batched sequence requires [batch, seq, hs]
        :param mask: binary mask with shape batch, seq_len, 1
        :param aggregate: max, avg, sum
        :param dim: The dimension to take the max. for batch, seq, hs it is 1
        :return:
    """

    seq_tensor_masked = seq_tensor * mask.unsqueeze(-1)
    aggr_func = None
    if aggregate == "last":
        seq = get_final_encoder_states(seq_tensor, mask, bidirectional)
    elif aggregate == "max":
        aggr_func = torch.max
        seq, _ = aggr_func(seq_tensor_masked, dim=dim)
    elif aggregate == "min":
        aggr_func = torch.min
        seq, _ = aggr_func(seq_tensor_masked, dim=dim)
    elif aggregate == "sum":
        aggr_func = torch.sum
        seq = aggr_func(seq_tensor_masked, dim=dim)
    elif aggregate == "avg":
        aggr_func = torch.sum
        seq = aggr_func(seq_tensor_masked, dim=dim)
        seq_lens = torch.sum(mask, dim=dim)  # this returns batch_size, 1
        seq = seq / seq_lens.view([-1, 1])

    return seq


def aggregate_and_copy_context_tokens_values_to_target_vocab(sequence_token_ids,
                                                             sequence_token_probs,
                                                             vocab_size):
    """
    Copy mechanism to aggregate the context tokens values to a target vocab.
    :param sequence_token_ids: Sequence tokens ids. Size: batch_size, max_sequence_len
    :param sequence_token_probs: Sequence tokens probabilities. Size: batch_size, max_sequence_len
    :param vocab_size: Vocab probabilites. Size: batch_size, vocab_size
    :return:
    """
    # Updates the probabilities to the global vocabulary
    # We are brave and we are going to use the *experimental* sparse API:
    # See example of this at https://pytorch.org/docs/stable/sparse.html

    batch_size = sequence_token_ids.shape[0]

    # size: batch_size
    index_rows = torch.arange(batch_size)
    if sequence_token_probs.is_cuda:
        index_rows = index_rows.cuda()

    # size: batch_size * max_sequence_len
    index_rows = index_rows.unsqueeze(-1).repeat(1, sequence_token_probs.shape[-1]).view(-1)

    # size: batch_size * max_sequence_len
    index_cols = sequence_token_ids.view(-1)

    # size: 2, batch_size * max_sequence_len
    index = torch.cat([index_rows.unsqueeze(0), index_cols.unsqueeze(0)], 0)
    values = sequence_token_probs.view(-1)

    # size: batch_size, vocab_size
    if torch.cuda.is_available():
        probs_to_vocab = torch.cuda.sparse.FloatTensor(index, values, torch.Size([batch_size, vocab_size])).to_dense()
    else:
        probs_to_vocab = torch.sparse.FloatTensor(index, values, torch.Size([batch_size, vocab_size])).to_dense()

    return probs_to_vocab


def batched_gather(values, index):
    """
    Gathers values batch-wise
    :param values: Batched values to be gathered. Size: batch_size, Any
    :param index: Batch-wise indices to be picked from `values`. Size: batch_size, Any
    :return: Picked values. Size: same as `index.shape`
    """
    values_flat = values.view(-1)
    batch_size = values.shape[0]
    values_cols = values.shape[1]

    index_offsets = torch.arange(batch_size).unsqueeze(-1)

    if index.is_cuda:
        values_cols = torch.tensor(np.array([values_cols])).long().cuda()
        index_offsets = index_offsets.cuda()
    index_new = index + index_offsets * values_cols

    # print("index_new.shape:%s" % str(index_new.shape))
    # print("values_flat.shape:%s" % str(values_flat.shape))
    # print("index.shape:%s" % str(index.shape))
    gathered_values = values_flat.gather(dim=0, index=index_new.view(-1)).view(index.shape)

    return gathered_values


def copy_attn_sum_to_target_vocab(token_sequence_probs, token_sequence_ids,
                                  tokens_pointers, tokens_pointers_mask,
                                  vocab_size,
                                  return_mask=False):
    """
    Copy token sequence probabilities to target tokens vocab. Size: batch_size, max_unique_tokens
    :param token_sequence_probs: Calculated token sequence probabilities
    :param token_sequence_ids: Input token sequence ids. Size: batch_size, max_unique_tokens
    :param tokens_pointers: Unique tokens pointers to the token_sequence. Size: batch_size, max_unique_tokens, max_occurence
    :param tokens_pointers_mask: Mask for the tokens pointers. Size: batch_size, max_unique_tokens, max_occurence
    :return:  Size: batch_size, vocab_size
    """

    batch_size = token_sequence_probs.shape[0]

    # Pick the values
    # size:  batch_size, max_unique_tokens, max_occurence
    picked_values = batched_gather(token_sequence_probs, tokens_pointers.view(batch_size, -1)) \
        .view(tokens_pointers.shape)
    picked_values = picked_values * tokens_pointers_mask


    # Calculate the value sum over the picked values
    # shape: batch_size, max_unique_tokens
    pointer_tokens_probs = picked_values.sum(-1)

    # We need the ids of tokens in vocabulary for each unique token
    # We gather unique token ids for each pointer. This is the token_id of the first occurence of
    # the token in the sequence.

    # size: batch_size, max_unique_tokens
    pointer_tokens_ids = batched_gather(token_sequence_ids, tokens_pointers[:, :, :1]
                                        .view(tokens_pointers.shape[0], tokens_pointers.shape[1])) \
                        .view(tokens_pointers.shape[:2])

    # Get mask for `pointer_tokens_ids` from `tokens_pointers_tensor_mask` which should be 1 rank higher
    # size: batch_size, max_unique_tokens
    pointer_tokens_ids_mask = tokens_pointers_mask[:, :, :1].view(pointer_tokens_ids.shape[0],
                                                                  pointer_tokens_ids.shape[1])
    pointer_tokens_ids = pointer_tokens_ids * pointer_tokens_ids_mask.long()

    # Putting the values in the TARGET VOCAB shape
    # Use sparse vector for building the batched target vocabulary
    i_rows = torch.arange(batch_size).unsqueeze(-1).repeat(1, pointer_tokens_probs.shape[-1]).view(-1)
    if pointer_tokens_ids.is_cuda:
        i_rows = i_rows.cuda()
    i_cols = pointer_tokens_ids.view(-1)
    i = torch.cat([i_rows.unsqueeze(0), i_cols.unsqueeze(0)], 0)

    # size: batch_size, max_unique_tokens
    v_context_ids = torch.arange(pointer_tokens_ids.shape[1]).unsqueeze(0).repeat(batch_size, 1)
    if pointer_tokens_ids.is_cuda:
        v_context_ids = v_context_ids.cuda()

    v_context_ids = v_context_ids * pointer_tokens_ids_mask.long()

    target_vocab_with_source_local_ids = torch.sparse.LongTensor(i, v_context_ids.view(-1), torch.Size([batch_size, vocab_size])).to_dense()

    # Gather the values from the sumed unique tokens probs in `pointer_tokens_probs`
    vocab_probs = batched_gather(pointer_tokens_probs, target_vocab_with_source_local_ids).view(target_vocab_with_source_local_ids.shape)
    vocab_probs_mask = (target_vocab_with_source_local_ids > 0).float()
    vocab_probs *= vocab_probs_mask

    if return_mask:
        return vocab_probs, vocab_probs_mask
    else:
        return vocab_probs


def copy_attn_sum_to_target_vocab_matmul(token_sequence_probs,
                                        token_sequence_ids,
                                        vocab_size):
    """
    Copy token sequence probabilities to target tokens vocab. Size: batch_size, max_unique_tokens
    :param token_sequence_probs: Calculated token sequence probabilities
    :param token_sequence_ids: Input token sequence ids. Size: batch_size, max_unique_tokens
    :return:  Size: batch_size, vocab_size
    """

    batch_size = token_sequence_probs.shape[0]
    max_enc_len = token_sequence_probs.shape[1]

    # map the input sequences to vocab ids
    token_sequence_ids_map = token_sequence_ids.unsqueeze(2).data  # [B, S, 1]
    seq_ids_one_hot = torch.zeros(batch_size, max_enc_len, vocab_size, requires_grad=False, dtype=torch.float32)
    if token_sequence_ids_map.is_cuda:
        seq_ids_one_hot = seq_ids_one_hot.cuda()
    seq_ids_one_hot.scatter_(2, token_sequence_ids_map, 1)  # [B, S, |V|]

    # sum probabilies
    seq_probs_to_vocab = torch.bmm(token_sequence_probs.unsqueeze(1), seq_ids_one_hot)  # [B, 1, |V|]
    seq_probs_to_vocab = seq_probs_to_vocab.squeeze()  # [B, |V|]

    return seq_probs_to_vocab

