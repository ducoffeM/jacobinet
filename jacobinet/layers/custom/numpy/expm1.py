from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Expm1  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardExpm1(BackwardNonLinearLayer):
    """
    Backward layer for the `Expm1` operation.
    Gradient: dy/dx = exp(x)
    """

    def __init__(
        self,
        layer: Expm1,
        **kwargs: Any,
    ):
        super().__init__(layer=layer, **kwargs)

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if input is None:
            raise ValueError("Input tensor is required for BackwardExpm1.")

        # Compute exp(x)
        local_grad = K.exp(input)

        # Chain rule: grad_output * exp(input)
        return gradient * local_grad


def get_backward_Expm1(layer: Expm1) -> Layer:
    """
    Creates a `BackwardExpm1` layer for the given `Expm1` layer.

    Parameters:
    - `layer`: An instance of the `Expm1` layer.

    Returns:
    - An instance of `BackwardExpm1` which computes grad_output * exp(input).
    """
    return BackwardExpm1(layer)
