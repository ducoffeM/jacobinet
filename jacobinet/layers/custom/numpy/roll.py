from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Roll  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardRoll(BackwardLinearLayer):
    use_W_numerically: bool = True
    """
    Implements the backward pass for keras_custom.layers.numpy.Roll.

    ### Forward:
        y = keras.ops.roll(x, shift, axis)

    ### Backward:
        dL/dx = keras.ops.roll(dL/dy, -shift, axis)
        Because `Roll` is a permutation (no element mixing),
        its inverse operation is simply a roll in the opposite direction.
    """


def get_backward_Roll(layer: Roll) -> Layer:
    """
    Creates a BackwardRoll layer corresponding to a given Roll layer.

    ### Example
    ```python
    from keras_custom.layers.numpy import Roll
    from keras_custom.backward import get_backward_Roll

    roll_layer = Roll(shift=2, axis=1)
    backward_layer = get_backward_Roll(roll_layer)

    grad_input = backward_layer(gradient_tensor)
    ```
    """
    return BackwardRoll(layer)
