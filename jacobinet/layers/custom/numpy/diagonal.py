from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer, BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Diagonal  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardDiagonal(BackwardLinearLayer):
    use_W_numerically: bool = True
    """
    Implements the backward pass for keras_custom.layers.numpy.Diagonal.

    ### Forward:
        y = ops.diagonal(x, offset, axis1, axis2)
        (output rank = input rank - 1)

    ### Backward:
        dL/dx = ops.diag_embed(dL/dy, offset, axis1, axis2)

    The backward pass "scatters" the incoming gradient along the
    diagonal positions of a zero tensor of the same shape as the input.
    """


def get_backward_Diagonal(layer: Diagonal) -> Layer:
    """
    Creates a BackwardDiagonal layer corresponding to a given Diagonal layer.

    ### Example
    ```python
    from keras_custom.layers.numpy import Diagonal
    from keras_custom.backward import get_backward_Diagonal

    diag_layer = Diagonal(offset=0, axis1=-2, axis2=-1)
    backward_layer = get_backward_Diagonal(diag_layer)

    grad_input = backward_layer(gradient_tensor)
    ```
    """
    return BackwardDiagonal(layer)
