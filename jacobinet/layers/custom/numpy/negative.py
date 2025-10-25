from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Negative  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardNegative(BackwardLinearLayer):
    """
    Backward layer for the `Negative` operation.
    Computes gradient: dy/dx = -1
    """

    def __init__(
        self,
        layer: Negative,
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
        # Local gradient of -x is -1
        return -gradient


def get_backward_Negative(layer: Negative) -> Layer:
    """
    Creates a `BackwardNegative` layer for the given `Negative` layer.

    ### Parameters:
    - `layer`: An instance of the `Negative` layer.

    ### Returns:
    - An instance of `BackwardNegative`, which computes -grad_output.
    """
    return BackwardNegative(layer)
