import logging

from allennlp.common import Registrable
from typing import Any, Dict, List, Optional

import torch
from torch.nn.functional import nll_loss

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("bidaf_interaction_encoder")
class BidafInteractionEncoder(torch.nn.Module, Registrable):
    """
    This class implements Minjoon Seo's `Bidirectional Attention Flow model
    <https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Seo-Kembhavi/7586b7cca1deba124af80609327395e613a20e9d>`_
    for answering reading comprehension questions (ICLR 2017).

    The basic layout is pretty simple: encode words as a combination of word embeddings and a
    character-level encoder, pass the word representations through a bi-LSTM/GRU, use a matrix of
    attentions to put question information into the passage word representations (this is the only
    part that is at all non-standard), pass this through another few layers of bi-LSTMs/GRUs, and
    do a softmax over span start and span end.

    Parameters
    ----------
    input_size : ``int``
        Input size.
    num_highway_layers : ``int``
        The number of highway layers to use in between embedding the input and passing it through
        the phrase layer.
    phrase_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and doing the bidirectional attention.
    similarity_function : ``SimilarityFunction``
        The similarity function that we will use when comparing encoded passage and question
        representations.
    modeling_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between the bidirectional
        attention and predicting span start and end.
    span_end_encoder : ``Seq2SeqEncoder``
        The encoder that we will use to incorporate span start predictions into the passage state
        before predicting span end.
    dropout : ``float``, optional (default=0.2)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    mask_lstms : ``bool``, optional (default=True)
        If ``False``, we will skip passing the mask to the LSTM layers.  This gives a ~2x speedup,
        with only a slight performance decrease, if any.  We haven't experimented much with this
        yet, but have confirmed that we still get very similar performance with much faster
        training times.  We still use the mask for all softmaxes, but avoid the shuffling that's
        required when using masking with pytorch LSTMs.
    """

    def __init__(self,
                 input_size: int,
                 num_highway_layers: int,
                 phrase_layer: Seq2SeqEncoder,
                 similarity_function: SimilarityFunction,
                 modeling_layer: Seq2SeqEncoder,
                 dropout: float = 0.2,
                 mask_lstms: bool = True) -> None:
        super(BidafInteractionEncoder, self).__init__()

        self._highway_layer = TimeDistributed(Highway(input_size,
                                                      num_highway_layers))
        self._phrase_layer = phrase_layer
        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        self._modeling_layer = modeling_layer

        encoding_dim = phrase_layer.get_output_dim()
        modeling_dim = modeling_layer.get_output_dim()

        # Bidaf has lots of layer dimensions which need to match up - these aren't necessarily
        # obvious from the configuration files, so we check here.
        check_dimensions_match(modeling_layer.get_input_dim(), 4 * encoding_dim,
                               "modeling layer input dim", "4 * encoding dim")
        check_dimensions_match(input_size, phrase_layer.get_input_dim(),
                               "input_size", "phrase layer input dim")
        
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms


    def forward(self,  # type: ignore
                question: torch.Tensor,
                passage: torch.Tensor,
                question_mask: torch.Tensor,
                passage_mask: torch.Tensor,
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : torch.Tensor
            Encoded question
        passage : torch.Tensor
            Encoded passage
        question_mask : torch.Tensor
            Encoded question mask
        passage_mask : torch.Tensor
            Encoded passage mask

        Returns
        -------
        An output dictionary consisting of:
        encoded_question : torch.Tensor
            Encoded question using the BiDAF interaction.
        encoded_passage : torch.Tensor
            Encoded passage using the BiDAF interaction.

        """
        embedded_question = self._highway_layer(question)
        embedded_passage = self._highway_layer(passage)
        batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        question_lstm_mask = question_mask if self._mask_lstms else None
        passage_lstm_mask = passage_mask if self._mask_lstms else None

        encoded_question = self._dropout(self._phrase_layer(embedded_question, question_lstm_mask))
        encoded_passage = self._dropout(self._phrase_layer(embedded_passage, passage_lstm_mask))
        encoding_dim = encoded_question.size(-1)

        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = util.masked_softmax(passage_question_similarity, question_mask)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = util.replace_masked_values(passage_question_similarity,
                                                       question_mask.unsqueeze(1),
                                                       -1e7)
        # Shape: (batch_size, passage_length)
        question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # Shape: (batch_size, passage_length)
        question_passage_attention = util.masked_softmax(question_passage_similarity, passage_mask)
        # Shape: (batch_size, encoding_dim)
        question_passage_vector = util.weighted_sum(encoded_passage, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(batch_size,
                                                                                    passage_length,
                                                                                    encoding_dim)

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat([encoded_passage,
                                          passage_question_vectors,
                                          encoded_passage * passage_question_vectors,
                                          encoded_passage * tiled_question_passage_vector],
                                         dim=-1)

        modeled_passage = self._dropout(self._modeling_layer(final_merged_passage, passage_lstm_mask))
        modeling_dim = modeled_passage.size(-1)

        return modeled_passage

