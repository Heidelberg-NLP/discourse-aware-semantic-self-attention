from allennlp.modules import Embedding
from overrides import overrides
import torch
from torch.nn import Dropout, Linear

from allennlp.nn.util import masked_softmax, weighted_sum
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


@Seq2SeqEncoder.register("multi_head_semantic_flat_self_attention")
class MultiHeadSemanticFlatSelfAttention(Seq2SeqEncoder):
    # pylint: disable=line-too-long
    """
    This class implements the key-value scaled dot product attention mechanism
    detailed in the paper `Attention is all you Need
    <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

    The attention mechanism is a weighted sum of a projection V of the inputs, with respect
    to the scaled, normalised dot product of Q and K, which are also both linear projections
    of the input. This procedure is repeated for each attention head, using different parameters.

    Parameters
    ----------
    num_heads : ``int``, required.
        The number of attention heads to use.
    input_dim : ``int``, required.
        The size of the last dimension of the input tensor.
    attention_dim ``int``, required.
        The total dimension of the query and key projections which comprise the
        dot product attention function. Must be divisible by ``num_heads``.
    values_dim : ``int``, required.
        The total dimension which the input is projected to for representing the values,
        which are combined using the attention. Must be divisible by ``num_heads``.
    output_projection_dim : ``int``, optional (default = None)
        The dimensionality of the final output projection. If this is not passed
        explicitly, the projection has size `input_size`.
    attention_dropout_prob : ``float``, optional (default = 0.1).
        The dropout probability applied to the normalised attention
        distributions.
    """
    def __init__(self,
                 num_heads: int,
                 input_dim: int,
                 attention_dim: int,
                 num_semantic_labels: int,
                 values_dim: int,
                 output_projection_dim: int = None,
                 attention_dropout_prob: float = 0.1,
                 use_semantic_views=True,
                 multi_head_attention_batch_computation=False,
                 use_separate_label_embeddings_for_q_and_k=True,
                 ) -> None:
        super(MultiHeadSemanticFlatSelfAttention, self).__init__()

        self.use_semantic_views = use_semantic_views
        self.use_separate_label_embeddings_for_q_and_k = use_separate_label_embeddings_for_q_and_k
        self.multi_head_attention_batch_computation = multi_head_attention_batch_computation

        self._single_head_attention_dim = int(attention_dim / num_heads)
        self._semantic_label_embedding_q_w = Embedding(num_embeddings=num_semantic_labels,
                                                       embedding_dim=input_dim * self._single_head_attention_dim)

        self._semantic_label_embedding_q_b = Embedding(num_embeddings=num_semantic_labels,
                                                       embedding_dim=self._single_head_attention_dim)

        self._semantic_label_embedding_k_w = Embedding(num_embeddings=num_semantic_labels,
                                                       embedding_dim=input_dim * self._single_head_attention_dim)

        self._semantic_label_embedding_k_b = Embedding(num_embeddings=num_semantic_labels,
                                                       embedding_dim=self._single_head_attention_dim)

        self._num_heads = num_heads
        self._input_dim = input_dim
        self._output_dim = output_projection_dim or input_dim
        self._attention_dim = attention_dim
        self._values_dim = values_dim

        if attention_dim % num_heads != 0:
            raise ValueError(f"Key size ({attention_dim}) must be divisible by the number of "
                             f"attention heads ({num_heads}).")

        if values_dim % num_heads != 0:
            raise ValueError(f"Value size ({values_dim}) must be divisible by the number of "
                             f"attention heads ({num_heads}).")

        self._combined_projection = Linear(input_dim, 2 * attention_dim + values_dim)

        self._values_projection = Linear(input_dim, values_dim)

        self._scale = (input_dim // num_heads) ** 0.5
        self._output_projection = Linear(values_dim, self._output_dim)
        self._attention_dropout = Dropout(attention_dropout_prob)

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return False

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                semantic_views_q: torch.Tensor,
                semantic_views_sent_mask: torch.Tensor,
                mask: torch.LongTensor = None) -> torch.FloatTensor:
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).

        Returns
        -------
        A tensor of shape (batch_size, timesteps, output_projection_dim),
        where output_projection_dim = input_dim by default.
        """
        num_heads = self._num_heads

        batch_size, timesteps, _ = inputs.size()
        if mask is None:
            mask = inputs.new_ones(batch_size, timesteps)

        if self.use_semantic_views:
            # Shape (batch_size, timesteps, 2 * attention_dim + values_dim)

            values = self._values_projection(inputs)

            # split by attention dim - if values_dim > attention_dim, we will get more
            # than 3 elements returned. All of the rest are the values vector, so we
            # just concatenate them back together again below.

            bs = inputs.shape[0]
            seq_len = inputs.shape[1]
            input_dim = inputs.shape[-1]

            similarities_per_head = []
            masks_per_head = []

            if not self.multi_head_attention_batch_computation:
                for head_slice in zip(torch.split(semantic_views_q, 1, 1), torch.split(semantic_views_sent_mask, 1, 1)):
                    semantic_views_q_head, semantic_views_sent_mask_head = head_slice
                    semantic_views_q_head = semantic_views_q_head.contiguous().squeeze(1)
                    semantic_views_sent_mask_head = semantic_views_sent_mask_head.contiguous()

                    semantic_views_sent_mask_head = semantic_views_sent_mask_head.repeat(1, seq_len, 1)
                    semantic_views_sent_mask_head = (semantic_views_sent_mask_head == semantic_views_sent_mask_head.transpose(2, 1)) * (semantic_views_sent_mask_head > 0) * (semantic_views_sent_mask_head > 0)

                    semantic_views_sent_mask_head = semantic_views_sent_mask_head.float()

                    mask_repeated = mask.unsqueeze(1).repeat(1, seq_len, 1)
                    semantic_views_sent_mask_head = semantic_views_sent_mask_head * mask_repeated

                    masks_per_head.append(semantic_views_sent_mask_head)

                    # q_head
                    q_head_w = self._semantic_label_embedding_q_w(semantic_views_q_head).view(
                                                                  list(semantic_views_q_head.shape)
                                                                   + [input_dim, -1])
                    #q_head = torch.einsum('bid,bidn->bin', [inputs,  q_head_w])
                    q_head = torch.bmm(inputs.view(bs * seq_len, 1, input_dim),  q_head_w.view(bs * seq_len, input_dim, -1))
                    q_head = q_head.view(bs, seq_len, -1)
                    q_head = q_head + self._semantic_label_embedding_q_b(semantic_views_q_head)

                    if self.use_separate_label_embeddings_for_q_and_k:
                        # q_head
                        k_head_w = self._semantic_label_embedding_q_w(semantic_views_q_head).view(
                                                                      list(semantic_views_q_head.shape)
                                                                       + [input_dim, -1])
                        #k_head = torch.einsum('bid,bidn->bin', [inputs,  k_head_w])
                        k_head = torch.bmm(inputs.view(bs * seq_len, 1, input_dim),  k_head_w.view(bs * seq_len, input_dim, -1))
                        k_head = k_head.view(bs, seq_len, -1)
                        k_head = k_head + self._semantic_label_embedding_q_b(semantic_views_q_head)
                    else:
                        k_head = q_head

                    att_head = torch.bmm(q_head / self._scale, k_head.transpose(1, 2))
                    att_head = att_head * semantic_views_sent_mask_head
                    att_head = att_head.view(bs, 1, seq_len, seq_len)

                    similarities_per_head.append(att_head)

                scaled_similarities = torch.cat(similarities_per_head, dim=1).view(bs * num_heads, seq_len, seq_len)

                # Shape (num_heads * batch_size, timesteps, values_dim / num_heads)
                values_per_head = values.view(batch_size, timesteps, num_heads, int(self._values_dim / num_heads))
                values_per_head = values_per_head.transpose(1, 2).contiguous()
                values_per_head = values_per_head.view(batch_size * num_heads, timesteps,
                                                       int(self._values_dim / num_heads))

                # shape (num_heads * batch_size, timesteps, timesteps)
                # Normalise the distributions, using the same mask for all heads.
                attention = masked_softmax(scaled_similarities,
                                           mask.repeat(1, num_heads).view(batch_size * num_heads, timesteps),
                                           memory_efficient=True)
            else:
                # Shape (bs, num_heads, seq_len, d)
                inputs_by_head = inputs.unsqueeze(1).repeat(1, self._num_heads, 1, 1)
                head_dim = self._single_head_attention_dim

                def get_input_per_head_using_sem_views(inputs_by_head, semantic_view, emb_w, emb_b):
                    semantic_veiws_by_head_w = emb_w(semantic_view).view([bs, num_heads, seq_len, input_dim, head_dim])
                    semantic_veiws_by_head_b = emb_b(semantic_view).view([bs, num_heads, seq_len, head_dim])

                    res = torch.bmm(inputs_by_head.view(bs * num_heads * seq_len, 1, input_dim),
                                    semantic_veiws_by_head_w.view(bs * num_heads * seq_len, input_dim, head_dim)) \
                                    .view(bs, num_heads, seq_len, head_dim) \
                          + semantic_veiws_by_head_b

                    return res

                queries_per_head = get_input_per_head_using_sem_views(inputs_by_head, semantic_views_q,
                                                                      self._semantic_label_embedding_q_w,
                                                                      self._semantic_label_embedding_q_b)
                queries_per_head = queries_per_head.view(batch_size * num_heads, timesteps, head_dim)

                keys_per_head = get_input_per_head_using_sem_views(inputs_by_head, semantic_views_q,
                                                                      self._semantic_label_embedding_k_w,
                                                                      self._semantic_label_embedding_k_b)
                keys_per_head = keys_per_head.view(batch_size * num_heads, timesteps, head_dim)

                # shape (num_heads * batch_size, timesteps, timesteps)
                scaled_similarities = torch.bmm(queries_per_head / self._scale, keys_per_head.transpose(1, 2))

                # mask
                # Shape (bs, num_heads, seq_len, seq_len)
                semantic_views_sent_mask_tokenwise = semantic_views_sent_mask.unsqueeze(2).repeat(1, 1, seq_len, 1)
                # allow only per-scope mask - like sentence-wise, neighbouring sentences, etc.
                semantic_views_sent_mask_tokenwise = (semantic_views_sent_mask_tokenwise == semantic_views_sent_mask_tokenwise.transpose(3, 2)) \
                                                     * (semantic_views_sent_mask_tokenwise > 0) # this multiplication is the masking of padded zeros!
                semantic_views_sent_mask_tokenwise = semantic_views_sent_mask_tokenwise.float()
                semantic_views_sent_mask_tokenwise = semantic_views_sent_mask_tokenwise\
                                                        .view(bs * num_heads, seq_len, seq_len)
                # masked the similarities
                scaled_similarities = scaled_similarities * semantic_views_sent_mask_tokenwise

                # Shape (num_heads * batch_size, timesteps, values_dim / num_heads)
                values_per_head = values.view(batch_size, timesteps, num_heads, int(self._values_dim / num_heads))
                values_per_head = values_per_head.transpose(1, 2).contiguous()
                values_per_head = values_per_head.view(batch_size * num_heads, timesteps, int(self._values_dim / num_heads))

                # shape (num_heads * batch_size, timesteps, timesteps)
                # Normalise the distributions, using the same mask for all heads.
                attention = masked_softmax(scaled_similarities,
                                           semantic_views_sent_mask_tokenwise,
                                           memory_efficient=True)

        else:
            # Shape (batch_size, timesteps, 2 * attention_dim + values_dim)
            combined_projection = self._combined_projection(inputs)
            # split by attention dim - if values_dim > attention_dim, we will get more
            # than 3 elements returned. All of the rest are the values vector, so we
            # just concatenate them back together again below.
            queries, keys, *values = combined_projection.split(self._attention_dim, -1)
            queries = queries.contiguous()
            keys = keys.contiguous()
            values = torch.cat(values, -1).contiguous()

            # Shape (num_heads * batch_size, timesteps, values_dim / num_heads)
            values_per_head = values.view(batch_size, timesteps, num_heads, int(self._values_dim/num_heads))
            values_per_head = values_per_head.transpose(1, 2).contiguous()
            values_per_head = values_per_head.view(batch_size * num_heads, timesteps, int(self._values_dim/num_heads))

            # Shape (num_heads * batch_size, timesteps, attention_dim / num_heads)
            queries_per_head = queries.view(batch_size, timesteps, num_heads, int(self._attention_dim/num_heads))
            queries_per_head = queries_per_head.transpose(1, 2).contiguous()
            queries_per_head = queries_per_head.view(batch_size * num_heads, timesteps, int(self._attention_dim/num_heads))

            # Shape (num_heads * batch_size, timesteps, attention_dim / num_heads)
            keys_per_head = keys.view(batch_size, timesteps, num_heads, int(self._attention_dim/num_heads))
            keys_per_head = keys_per_head.transpose(1, 2).contiguous()
            keys_per_head = keys_per_head.view(batch_size * num_heads, timesteps, int(self._attention_dim/num_heads))

            # shape (num_heads * batch_size, timesteps, timesteps)
            scaled_similarities = torch.bmm(queries_per_head / self._scale, keys_per_head.transpose(1, 2))

            # shape (num_heads * batch_size, timesteps, timesteps)
            # Normalise the distributions, using the same mask for all heads.
            attention = masked_softmax(scaled_similarities,
                                       mask.repeat(1, num_heads).view(batch_size * num_heads, timesteps),
                                       memory_efficient=True)

        attention = self._attention_dropout(attention)

        # Take a weighted sum of the values with respect to the attention
        # distributions for each element in the num_heads * batch_size dimension.
        # shape (num_heads * batch_size, timesteps, values_dim/num_heads)
        outputs = weighted_sum(values_per_head, attention)

        # Reshape back to original shape (batch_size, timesteps, values_dim)
        # shape (batch_size, num_heads, timesteps, values_dim/num_heads)
        outputs = outputs.view(batch_size, num_heads, timesteps, int(self._values_dim / num_heads))
        # shape (batch_size, timesteps, num_heads, values_dim/num_heads)
        outputs = outputs.transpose(1, 2).contiguous()
        # shape (batch_size, timesteps, values_dim)
        outputs = outputs.view(batch_size, timesteps, self._values_dim)

        # Project back to original input size.
        # shape (batch_size, timesteps, input_size)
        outputs = self._output_projection(outputs)
        return outputs
