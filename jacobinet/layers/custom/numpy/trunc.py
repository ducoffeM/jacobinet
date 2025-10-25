from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Trunc  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardTrunc(BackwardNonLinearLayer):
    """
    Implements the backward pass for keras_custom.layers.numpy.Trunc.

    ### Forward:
        y = ops.trunc(x)
    ### Backward:
        dL/dx = 0  (since truncation is piecewise constant)

    The gradient of truncation is zero everywhere.
    """

    def __init__(self, layer: Trunc, **kwargs: Any):
        super().__init__(layer=layer, **kwargs)

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if input is None:
            raise ValueError("Input tensor is required for BackwardTrunc.")
        if gradient is None:
            raise ValueError("Gradient tensor is required for BackwardTrunc.")

        # The derivative of trunc(x) w.r.t. x is zero everywhere
        grad_input = K.zeros_like(input)
        return grad_input


def get_backward_Trunc(layer: Trunc) -> Layer:
    """
    Creates a BackwardTrunc layer corresponding to a given Trunc layer.

    ### Example:
    ```python
    from keras_custom.layers.numpy import Trunc
    from keras_custom.backward import get_backward_Trunc

    trunc_layer = Trunc()
    backward_layer = get_backward_Trunc(trunc_layer)

    grad_input = backward_layer(gradient_tensor)
    ```
    """
    return BackwardTrunc(layer)
