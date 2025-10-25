from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Sin  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardSin(BackwardNonLinearLayer):
    """
    Backward layer for the `Sin` operation.
    Computes gradient: dy/dx = cos(x)
    """

    def __init__(
        self,
        layer: Sin,
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
            raise ValueError("Input tensor is required for BackwardSin.")

        # Compute local gradient: cos(x)
        local_grad = K.cos(input)

        # Chain rule
        return gradient * local_grad


def get_backward_Sin(layer: Sin) -> Layer:
    """
    Creates a `BackwardSin` layer for the given `Sin` layer.

    ### Parameters:
    - `layer`: An instance of the `Sin` layer.

    ### Returns:
    - An instance of `BackwardSin`, which computes grad_output * cos(input)
    """
    return BackwardSin(layer)
