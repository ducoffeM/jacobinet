from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Tril, Triu  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardTri(BackwardLinearLayer):
    use_W_numerically: bool = True
    """
    Implements the backward pass for keras_custom.layers.numpy.Tri(u/l).
    """


BackwardTril = BackwardTri
BackwardTriu = BackwardTri


def get_backward_Triu(layer: Triu) -> Layer:
    """
    Creates a BackwardDiagonal layer corresponding to a given Triu layer.

    ### Example
    ```python
    from keras_custom.layers.numpy import Triu
    from keras_custom.backward import get_backward_Triu

    diag_layer = Triu()
    backward_layer = get_backward_Triu(diag_layer)

    grad_input = backward_layer(gradient_tensor)
    ```
    """
    return BackwardTriu(layer)


def get_backward_Tril(layer: Tril) -> Layer:
    """
    Creates a BackwardDiagonal layer corresponding to a given Tril layer.

    ### Example
    ```python
    from keras_custom.layers.numpy import Tril
    from keras_custom.backward import get_backward_Tril

    diag_layer = Tril()
    backward_layer = get_backward_Tril(diag_layer)

    grad_input = backward_layer(gradient_tensor)
    ```
    """
    return BackwardTril(layer)
