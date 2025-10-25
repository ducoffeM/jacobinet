from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Repeat  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardRepeat(BackwardLinearLayer):
    use_W_numerically: bool = True
    """
    Implements the backward pass for keras_custom.layers.numpy.Repeat.

    ### Forward:
        y = keras.ops.repeat(x, repeats, axis)

    ### Backward:
        dL/dx = reduce_sum(reshape(dL/dy, [ ..., repeats, ... ]), axis=axis+1)
        Each input position receives the sum of the gradients from all
        its repeated copies along the `axis`.
    """


def get_backward_Repeat(layer: Repeat) -> Layer:
    """
    Creates a BackwardRepeat layer corresponding to a given Repeat layer.

    ### Example
    ```python
    from keras_custom.layers.numpy import Repeat
    from keras_custom.backward import get_backward_Repeat

    repeat_layer = Repeat(repeats=3, axis=1)
    backward_layer = get_backward_Repeat(repeat_layer)

    grad_input = backward_layer(gradient_tensor)
    ```
    """
    return BackwardRepeat(layer)
