from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Arctan  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardArctan(BackwardNonLinearLayer):
    """
    Implements the backward pass of the Arctan function.

    Given upstream gradient, multiplies by derivative:
    1 / (1 + x^2)

    Parameters:
    - `layer`: A Keras `Arctan` layer instance.

    Returns:
    - Gradient tensor w.r.t. input.
    """

    def __init__(
        self,
        layer: Arctan,
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
        return gradient * (1 / (1 + K.square(input)))


def get_backward_Arctan(layer: Arctan) -> Layer:
    """
    Factory function for BackwardArctan layer from an Arctan layer.
    """
    return BackwardArctan(layer)
