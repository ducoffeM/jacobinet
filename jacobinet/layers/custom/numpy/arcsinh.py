from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Arcsinh  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardArcsinh(BackwardNonLinearLayer):
    """
    Implements the backward pass of the Arcsinh function.

    Given upstream gradient, multiplies by derivative:
    1 / sqrt(1 + x^2)

    Parameters:
    - `layer`: A Keras `Arcsinh` layer instance.

    Returns:
    - Gradient tensor w.r.t. input.
    """

    def __init__(
        self,
        layer: Arcsinh,
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
        return gradient * (1 / K.sqrt(1 + K.square(input)))


def get_backward_Arcsinh(layer: Arcsinh) -> Layer:
    """
    Factory function for BackwardArcsinh layer from an Arcsinh layer.
    """
    return BackwardArcsinh(layer)
