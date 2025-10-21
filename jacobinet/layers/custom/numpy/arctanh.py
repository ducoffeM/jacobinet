from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Arctanh  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardArctanh(BackwardNonLinearLayer):
    """
    Implements the backward pass of the Arctanh function.

    Given upstream gradient, multiplies by derivative:
    1 / (1 - x^2)

    Parameters:
    - `layer`: A Keras `Arctanh` layer instance.

    Returns:
    - Gradient tensor w.r.t. input.
    """

    def __init__(
        self,
        layer: Arctanh,
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
        # Use K.maximum with epsilon for numerical stability near boundaries
        denominator = K.maximum(1 - K.square(input), keras.backend.epsilon())
        return gradient * (1 / denominator)


def get_backward_Arctanh(layer: Arctanh) -> Layer:
    """
    Factory function for BackwardArctanh layer from an Arctanh layer.
    """
    return BackwardArctanh(layer)
