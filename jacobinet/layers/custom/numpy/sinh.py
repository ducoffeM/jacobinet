from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Sinh  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardSinh(BackwardNonLinearLayer):
    """
    Backward layer for the `Sinh` operation.
    Computes gradient: dy/dx = cosh(x)
    """

    def __init__(
        self,
        layer: Sinh,
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
            raise ValueError("Input tensor is required for BackwardSinh.")

        # Local gradient: cosh(x)
        local_grad = K.cosh(input)

        # Chain rule: grad_output * local_grad
        return gradient * local_grad


def get_backward_Sinh(layer: Sinh) -> Layer:
    """
    Creates a `BackwardSinh` layer for the given `Sinh` layer.

    ### Parameters:
    - `layer`: An instance of the `Sinh` layer.

    ### Returns:
    - An instance of `BackwardSinh`, which computes grad_output * cosh(input)
    """
    return BackwardSinh(layer)
