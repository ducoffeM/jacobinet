from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Reciprocal  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardReciprocal(BackwardNonLinearLayer):
    """
    Backward layer for the `Reciprocal` operation (1 / x).
    Computes gradient: dy/dx = -1 / x^2
    """

    def __init__(
        self,
        layer: Reciprocal,
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
            raise ValueError("Input tensor is required for BackwardReciprocal.")

        # Compute local gradient: -1 / x^2
        safe_input = K.where(input == 0, keras.backend.epsilon(), input)
        local_grad = -1.0 / (safe_input**2)

        # Chain rule: grad_output * local_grad
        return gradient * local_grad


def get_backward_Reciprocal(layer: Reciprocal) -> Layer:
    """
    Creates a `BackwardReciprocal` layer for the given `Reciprocal` layer.

    ### Parameters:
    - `layer`: An instance of the `Reciprocal` layer.

    ### Returns:
    - An instance of `BackwardReciprocal`, which applies dy/dx = -1 / x^2.
    """
    return BackwardReciprocal(layer)
