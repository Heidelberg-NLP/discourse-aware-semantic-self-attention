import numpy as np
import time
import torch

from docqa.nn.util import aggregate_and_copy_context_tokens_values_to_target_vocab, batched_gather
from docqa.utils.processing_utils import get_token_lookup_pointers


def build_vocab(batched_tokens, lowercase=True):
    vocab = {}
    for tokens in batched_tokens:
        for token in tokens:
            if lowercase:
                token = token.lower()
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def pad(seq, pad_value, to_size):
    pad_seq = []
    if len(seq) > to_size:
        pad_seq = seq[:to_size]
    else:
        pad_seq = seq[:] + [pad_value] * (to_size - len(seq))
    return pad_seq


def tokens_to_ids(batched_tokens, vocab, unk_token_id=0):
    token_ids_batch = []
    for tokens in batched_tokens:
        curr_token_ids = []
        for token in tokens:
            if lowercase:
                token = token.lower()

            token_id = vocab.get(token, unk_token_id)
            curr_token_ids.append(token_id)
        token_ids_batch.append(curr_token_ids)

    return token_ids_batch


def pad_data_and_return_seqlens(data, pad_value=0):
    batch_data_seqlens = np.asarray([len(a) for a in data])
    max_len = max(batch_data_seqlens)
    batch_data_padded_x = np.asarray([pad(a, pad_value, max_len) for a in data])
    return batch_data_padded_x, batch_data_seqlens


def pad_and_get_seq_len_and_mask_rank3(data, pad_value=0, max_len_dim2=0):
    # data is of rank 3: [batch_size, number_tokens, number_deps]

    # number tokens
    batch_data_seqlens = [len(a) for a in data]
    max_len_seq = max(batch_data_seqlens)

    # max number of deps
    max_sub_length = 0
    if max_len_dim2 > 0:
        max_sub_length = max_len_dim2
    else:
        for item in data:
            # print "item:%s"%item
            for token in item:
                if len(token) > max_sub_length:
                    max_sub_length = len(token)

    # Build the input
    data_padded = []
    mask = []
    data_padded_seqlens = []
    for item in data:
        item_padded = []
        item_mask = []
        item_seqlens = []
        for token_deps in item:

            if len(token_deps) > max_sub_length:
                token_deps_padded = token_deps[:max_sub_length]
            else:
                token_deps_padded = token_deps + [pad_value]*(max_sub_length-len(token_deps))

            item_padded.append(token_deps_padded)

            token_deps_mask = [1]*len(token_deps)+[0]*(max_sub_length-len(token_deps))
            item_mask.append(token_deps_mask)
            item_seqlens.append(max(1, sum(token_deps_mask)))

        item_padded = item_padded + (max_len_seq-len(item_padded))*[[pad_value]*max_sub_length]
        data_padded.append(item_padded)

        item_mask = item_mask + (max_len_seq - len(item_mask)) * [[pad_value] * max_sub_length]
        mask.append(item_mask)

        item_seqlens = item_seqlens + (max_len_seq - len(item_mask)) * [1]
        data_padded_seqlens.append(item_seqlens)

        data_padded_seqlens_padded, _ = pad_data_and_return_seqlens(data_padded_seqlens)

    return np.asarray(data_padded), np.asarray(data_padded_seqlens_padded), np.asarray(mask)

def get_tokens_batch_mock():
    tokens_batch = 16 * [
        ['@start@', '@STORYSTART@', 'The', 'location', 'is', 'ENTERPRISE', ',', 'OVER', 'ROMULUS', ',', 'SPACE', '.',
         ' ', 'The', 'Enterprise', 'is', 'in', 'orbit', 'around', 'Romulus', '.', ' ', 'Remus', 'can', 'be', 'in',
         'the', 'distance', '.', '\n', 'PICARD', '(', 'V.O.', ')', ':', 'Captain', "'s", 'Log', '.', ' ', 'Stardate',
         '47844.9', '.', 'The', 'Enterprise', 'has', 'arrived', 'at', 'Romulus', 'and', 'is', 'waiting', 'at', 'the',
         'designated', 'coordinates', '.', ' ', 'All', 'our', 'hails', 'have', 'gone', 'unanswered', '.', ' ', 'We',
         "'ve", 'been', 'waiting', 'for', 'seventeen', 'hours', '.', '@SKIP@', 'of', 'us', '.', 'DATA', '\n ', 'Yes',
         ',', 'sir', '.', 'PICARD', '\n ', 'We', "'ll", 'find', 'a', 'way', 'off', 'together', '.', 'Recommendations',
         '?', 'DATA', '\n ', 'There', 'is', 'a', 'shuttlebay', '948', 'meters', 'from', 'our', 'current', 'location',
         '.', 'Data', 'inserts', 'the', 'ETU', 'back', 'into', 'his', 'wrist', 'and', 'they', 'leave', 't', 'chamber',
         '.', '@SKIP@', '\n ', 'For', 'how', 'long', '?', 'DATA', '\n ', 'Indefinitely', '.', 'B-9', '\n ', 'How',
         'long', 'is', 'that', '?', 'A', 'beat', '.', ' ', 'Data', 'gazes', 'at', 'the', 'B-9', 'deeply', '.', 'DATA',
         '\n ', 'A', 'long', 'time', ',', 'brother', '.', 'Data', 'reaches', 'forward', 'and', 'deactivates', 'his',
         'brother', '.', 'The', 'B-9', "'s", 'eyes', 'lose', 'the', 'spark', 'of', 'life', '.', ' ', 'He', 'stands',
         ',', 'frozen', 'Data', 'stands', 'before', 'him', 'Indefinitely', '.', 'B-9', '\n ', 'How', 'long', 'is',
         'that', '?', 'A', 'beat', '.', ' ', 'Data', 'gazes', 'at', 'the', 'B-9', 'deeply', '.', 'DATA', '\n ', 'A',
         'long', 'time', ',', 'brother', '.', 'Data', 'reaches', 'forward', 'and', 'deactivates', 'his', 'brother', '.',
         'The', 'B-9', "'s", 'eyes', 'lose', 'the', 'spark', 'of', 'life', '.', ' ', 'He', 'stands', ',', 'frozen',
         'Data', 'stands', 'before', 'him', '.', 'Indefinitely', '.', 'B-9', '\n ', 'How', 'long', 'is', 'that', '?',
         'A', 'beat', '.', ' ', 'Data', 'gazes', 'at', 'the', 'B-9', 'deeply', '.', 'DATA', '\n ', 'A', 'long', 'time',
         ',', 'brother', '.', 'Data', 'reaches', 'forward', 'and', 'deactivates', 'his', 'brother', '.', 'The', 'B-9',
         "'s", 'eyes', 'lose', 'the', 'spark', 'of', 'life', '.', ' ', 'He', 'stands', ',', 'frozen', 'Data', 'stands',
         'before', 'him', '.', 'Indefinitely', '.', 'B-9', '\n ', 'How', 'long', 'is', 'that', '?', 'A', 'beat', '.',
         ' ', 'Data', 'gazes', 'at', 'the', 'B-9', 'deeply', '.', 'DATA', '\n ', 'A', 'long', 'time', ',', 'brother',
         '.', 'Data', 'reaches', 'forward', 'and', 'deactivates', 'his', 'brother', '.', 'The', 'B-9', "'s", 'eyes',
         'lose', 'the', 'spark', 'of', 'life', '.', ' ', 'He', 'stands', ',', 'frozen', 'Data', 'stands', 'before',
         'him', '.', '.', '@SKIP@', '@end@']]
    tokens_batch = tokens_batch + 16 * [
        ['@start@', '@STORYSTART@', 'Captain', "'s", 'Log', '.', ' ', 'Stardate', '47844.9', '.', 'The', 'Enterprise',
         'has', 'arrived', 'at', 'Romulus', 'and', 'is', 'waiting', 'at', 'the', 'designated', 'coordinates', '.', ' ',
         'All', 'our', 'hails', 'have', 'gone', 'unanswered', '.', ' ', 'We', "'ve", 'been', 'waiting', 'for',
         'seventeen', 'hours', '.', '@SKIP@', 'of', 'us', '.', 'DATA', '\n ', 'Yes', ',', 'sir', '.', 'PICARD', '\n ',
         'We', "'ll", 'find', 'a', 'way', 'off', 'together', '.', 'Recommendations', '?', 'DATA', '\n ', 'There', 'is',
         'a', 'shuttlebay', '948', 'meters', 'from', 'our', 'current', 'location', '.', 'Data', 'inserts', 'the', 'ETU',
         'back', 'into', 'his', 'wrist', 'and', 'they', 'leave', 't', 'chamber', '.', '@SKIP@', '\n ', 'For', 'how',
         'long', '?', 'DATA', '\n ', 'Indefinitely', '.', 'B-9', '\n ', 'How', 'long', 'is', 'that', '?', 'A', 'beat',
         '.', ' ', 'Data', 'gazes', 'at', 'the', 'B-9', 'deeply', '.', 'DATA', '\n ', 'A', 'long', 'time', ',',
         'brother', '.', 'Data', 'reaches', 'forward', 'and', 'deactivates', 'his', 'brother', '.', 'The', 'B-9', "'s",
         'eyes', 'lose', 'the', 'spark', 'of', 'life', '.', ' ', 'He', 'stands', ',', 'frozen', 'Data', 'stands',
         'before', 'him', '.', '@SKIP@', '@end@']]

    return tokens_batch


if __name__ == "__main__":
    # With this code we want to implement an efficient copy mechanism from https://arxiv.org/pdf/1704.04368.pdf

    ##############
    # Test data ##
    ##############
    token_sequences_batch = get_tokens_batch_mock()
    batch_size = len(token_sequences_batch)
    lowercase = True

    tokens_vocab = build_vocab(token_sequences_batch, lowercase)
    max_sequence_length = max([len(x) for x in token_sequences_batch])

    vocab_size = 50000  # This is a dummy size > the actual size - used only for the computations - to simulate updating

    ##################
    # Pre-processing #
    ##################
    # Build a batch of token pointers
    start = time.time()
    unique_tokens_pointer_batch = []
    unique_tokens_batch = []
    for batch_id, tokens in enumerate(token_sequences_batch):
        unique_tokens, unique_tokens_pointer, pointer_lengths = get_token_lookup_pointers(tokens, lowercase)
        if batch_id % 10 == 0:
            print(unique_tokens_pointer)

        unique_tokens_pointer_batch.append(unique_tokens_pointer)
        unique_tokens_batch.append(unique_tokens)

    print("batch pointers generation done in %s s" % (time.time() - start))

    # Pad the pointers
    unique_tokens_pointer_batch_padded, pointers_padded_lens, pointers_mask = pad_and_get_seq_len_and_mask_rank3(
                                                                                unique_tokens_pointer_batch, 0)

    # Convert tokens to ids
    token_ids_batch = tokens_to_ids(token_sequences_batch, tokens_vocab)
    token_ids_padded, seq_lens = pad_data_and_return_seqlens(token_ids_batch)

    # Convert unique tokens to ids
    unique_tokens_ids_batch = tokens_to_ids(unique_tokens_batch, tokens_vocab)
    unique_tokens_ids_padded, unique_tokens_ids_lens = pad_data_and_return_seqlens(unique_tokens_ids_batch)

    #########
    # TORCH #
    #########

    # INPUT data

    # size: batch_size, max_seq_len
    tokens_ids_input_tensor = torch.from_numpy(token_ids_padded)
    bs, l = tokens_ids_input_tensor.shape

    # Pointers
    # size: batch_size, max_unique_tokens, max_occurence
    tokens_pointers_tensor = torch.from_numpy(unique_tokens_pointer_batch_padded)

    # Pointers mask
    # size: batch_size, max_unique_tokens
    tokens_pointers_tensor_lens = torch.from_numpy(pointers_padded_lens)
    lens = tokens_pointers_tensor_lens.view(-1)
    pointer_max_len, _ = lens.max(0)
    lens_to_mask = torch.arange(pointer_max_len).expand(len(lens), pointer_max_len) < lens.unsqueeze(1)

    # size: batch_size, max_unique_tokens, max_occurence
    tokens_pointers_tensor_mask = lens_to_mask.view(tokens_pointers_tensor.shape).float()

    # Gather values from probabilities

    # We assume that we already have the sequence tokens probabilities!
    # size: batch_size, max_seq_len
    batched_sequence_token_probs = torch.rand(batch_size, max_sequence_length, requires_grad=True)

    # Pick the values
    # size:  size: batch_size, max_unique_tokens, max_occurence
    picked_values = batched_gather(batched_sequence_token_probs, tokens_pointers_tensor.view(bs, -1))\
                                  .view(tokens_pointers_tensor.shape)
    #picked_values = batched_sequence_token_probs.gather(dim=1, index=tokens_pointers_tensor.view(bs, -1)).view(tokens_pointers_tensor.shape)
    picked_values = picked_values * tokens_pointers_tensor_mask

    # shape: batch_size, max_unique_tokens
    pointer_tokens_probs = picked_values.sum(-1)

    # Update vocab values using sparse tensor operation

    # We need the ids of tokens in vocabulary for each unique token
    # Option A) We can pass the actual ids of the unique tokens from `unique_tokens_batch`.
    # Option B) Gather unique token ids for each pointer. This is the token_id of the first occurence of
    # the token in the sequence.

    # A) Pass `unique_tokens_ids_padded` as parameter
    # size:  size: batch_size, max_unique_tokens
    pointer_tokens_ids = torch.from_numpy(unique_tokens_ids_padded)

    # B) Gather from the text
    # size:  size: batch_size, max_unique_tokens
    pointer_tokens_ids = tokens_ids_input_tensor.gather(dim=1, index=tokens_pointers_tensor[:, :, :1]
                                                        .view(tokens_pointers_tensor.shape[0], tokens_pointers_tensor.shape[1]))

    pointer_tokens_ids = batched_gather(tokens_ids_input_tensor, tokens_pointers_tensor[:, :, :1]
                                                        .view(tokens_pointers_tensor.shape[0], tokens_pointers_tensor.shape[1]))\
                                  .view(tokens_pointers_tensor.shape[:2])

    # Get mask for `pointer_tokens_ids` from `tokens_pointers_tensor_mask` which should be 1 rank higher
    # size:  size: batch_size, max_unique_tokens
    pointer_tokens_ids_mask = tokens_pointers_tensor_mask[:, :, :1].view(pointer_tokens_ids.shape[0],
                                                                         pointer_tokens_ids.shape[1])
    pointer_tokens_ids = pointer_tokens_ids * pointer_tokens_ids_mask.long()

    # Use sparse vector for building the batched targed vocabulary
    i_rows = torch.arange(batch_size).unsqueeze(-1).repeat(1, pointer_tokens_probs.shape[-1]).view(-1)
    i_cols = pointer_tokens_ids.view(-1)
    i = torch.cat([i_rows.unsqueeze(0), i_cols.unsqueeze(0)], 0)
    v_context_ids = torch.arange(pointer_tokens_ids.shape[1]).unsqueeze(0).repeat(batch_size, 1) * pointer_tokens_ids_mask.long()
    val_ids = torch.sparse.LongTensor(i, v_context_ids.view(-1), torch.Size([batch_size, vocab_size])).to_dense()
    #pointer_tokens_probs_copy = pointer_tokens_probs.copy(require_grad=True)
    #vocab_probs[i_rows, i_cols] += pointer_tokens_probs.view(-1)

    vocab_probs = batched_gather(pointer_tokens_probs, val_ids).view(val_ids.shape)
    vocab_probs_mask = (val_ids > 0).float()
    vocab_probs *= vocab_probs_mask
    print("vocab_probs.requires_grad=%s" % vocab_probs.requires_grad)

    # Update the probabilities to the global vocabulary
    # We are brave and we are going to use the *experimental* sparse API:
    # See example of this at https://pytorch.org/docs/stable/sparse.html
    i_rows = torch.arange(batch_size).unsqueeze(-1).repeat(1, pointer_tokens_probs.shape[-1]).view(-1)
    i_cols = pointer_tokens_ids.view(-1)
    i = torch.cat([i_rows.unsqueeze(0), i_cols.unsqueeze(0)], 0)
    v = pointer_tokens_probs.view(-1)

    # size: batch_size, vocab_size
    copy_probs = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, vocab_size]))
    copy_probs = copy_probs.to_dense()

    print(copy_probs.grad)


    ########################
    # SUPER SIMPLE VERSION #
    ########################

    # calculate probabilities without these operations
    # Update the probabilities to the global vocabulary
    # We are brave and we are going to use the *experimental* sparse API:
    # See example of this at https://pytorch.org/docs/stable/sparse.html
    i_rows = torch.arange(batch_size).unsqueeze(-1).repeat(1, batched_sequence_token_probs.shape[-1]).view(-1)
    i_cols = tokens_ids_input_tensor.view(-1)
    i = torch.cat([i_rows.unsqueeze(0), i_cols.unsqueeze(0)], 0)
    v = batched_sequence_token_probs.view(-1)

    print("v.requires_grad=%s" % v.requires_grad)
    # size: batch_size, vocab_size
    copy_probs_2 = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, vocab_size]))
    print("copy_probs_2.requires_grad=%s" % copy_probs_2.requires_grad)
    copy_probs_2 = copy_probs_2.to_dense()
    copy_probs_2 = torch.zeros((batch_size, vocab_size), requires_grad=True) + copy_probs_2

    print(torch.all(torch.lt(copy_probs, copy_probs_2 + 0.01)))

    copy_probs_3 = aggregate_and_copy_context_tokens_values_to_target_vocab(tokens_ids_input_tensor,
                                                                            batched_sequence_token_probs,
                                                                            vocab_size)
    print(torch.all(torch.lt(copy_probs, copy_probs_3+0.01)))
    print(copy_probs[0])












