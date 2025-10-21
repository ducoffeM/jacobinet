from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Arcsin  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardArcsin(BackwardNonLinearLayer):
    """
    Implements the backward pass of the Arcsin function.

    Given upstream gradient, multiplies by derivative:
    1 / sqrt(1 - x^2)

    Parameters:
    - `layer`: A Keras `Arcsin` layer instance.

    Returns:
    - Gradient tensor w.r.t. input.
    """

    def __init__(
        self,
        layer: Arcsin,
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
        # Use K.maximum with K.epsilon() to avoid sqrt(0) or negative inside sqrt
        return gradient * (1 / K.sqrt(K.maximum(1 - K.square(input), keras.backend.epsilon())))


def get_backward_Arcsin(layer: Arcsin) -> Layer:
    """
    Factory function for BackwardArcsin layer from an Arcsin layer.
    """
    return BackwardArcsin(layer)
