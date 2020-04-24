import numpy as np
import torch
from allennlp.common import Registrable, Params
from allennlp.modules import SimilarityFunction
from allennlp.nn import Activation
from typing import List

from torch.nn import Parameter

from functools import reduce


@SimilarityFunction.register("constant_tri")
class ConstantTriParams(SimilarityFunction):
    """
    This function applies linear transformation for each of the input tensors and takes the sum.
    If output_dim is 0, the dimensions of tensor_1_dim and tensor_2_dim of the two input tensors are expected
    to be equal and the output_dim is set to be their size. This is used since we might want to automatically infer
    the size of the output layer from automatically set values for tensor1 without explicitly knowing the semantic of
    the similarity function.

    Then the output is `W1x + W2y`` where W1 and W2 are linear transformation matrices.

    Parameters
    ----------
    tensor_1_dim : ``int``
        The dimension of the first tensor, ``x``, described above.  This is ``x.size()[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    tensor_2_dim : ``int``
        The dimension of the second tensor, ``y``, described above.  This is ``y.size()[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    tensor_3_dim : ``int``
        The dimension of the second tensor, ``y``, described above.  This is ``y.size()[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    output_dim : ``int``
        The dimension of the output tensor.
    activation : ``Activation``, optional (default=linear (i.e. no activation))
        An activation function applied after the ``w^T * [x;y] + b`` calculation.  Default is no
        activation.
    """
    def __init__(self,
                 tensor_1_dim: int,
                 tensor_2_dim: int,
                 tensor_3_dim: int,
                 output_constant: List[float],
                 ):

        super().__init__()

        output_constant = np.array(output_constant)
        self._output_constant = torch.tensor(output_constant, requires_grad=False, dtype=torch.float32)
        self._output_dim = self._output_constant.shape[-1]


    def forward(self, tensor1:torch.LongTensor, tensor2:torch.LongTensor, tensor3:torch.LongTensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        """
        Takes two tensors of the same shape, such as ``(batch_size, length_1, length_2,
        embedding_dim)``.  Transforms both tensor to a target output dimensions and returns a sum tensor with same
        number of dimensions, such as ``(batch_size, length, out_dim)``.
        """

        tile_size = reduce(lambda x, y: x * y, tensor1.shape[:-1])

        res_repr = self._output_constant.unsqueeze(0).repeat(tile_size, 1)

        return res_repr




