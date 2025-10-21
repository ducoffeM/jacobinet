from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Arccos  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardArccos(BackwardNonLinearLayer):
    """
    This layer implements the backward pass of the Arccos function.

    Given a gradient flowing from upstream, it multiplies it by the derivative
    of arccos(x): -1 / sqrt(1 - x^2)

    ### Parameters:
    - `layer`: A Keras `Arccos` layer instance.

    ### Returns:
    - A tensor representing the gradient of the loss w.r.t. the input.
    """

    def __init__(
        self,
        layer: Arccos,
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
        return gradient * (-1 / K.sqrt(1 - K.square(input)))


def get_backward_Arccos(layer: Arccos) -> Layer:
    """
    Factory function to create a BackwardArccos layer from an Arccos layer.

    ### Parameters:
    - `layer`: A Keras `Arccos` layer instance.

    ### Returns:
    - A `BackwardArccos` layer instance.
    """
    return BackwardArccos(layer)
