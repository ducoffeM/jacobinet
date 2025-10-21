from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Arccosh  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardArccosh(BackwardNonLinearLayer):
    """
    This layer implements the backward pass of the Arccosh function.

    Given a gradient flowing from upstream, it multiplies it by the derivative
    of arccosh(x): 1 / sqrt(x^2 - 1)

    ### Parameters:
    - `layer`: A Keras `Arccosh` layer instance.

    ### Returns:
    - A tensor representing the gradient of the loss w.r.t. the input.
    """

    def __init__(
        self,
        layer: Arccosh,
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
        return gradient * (1 / K.sqrt(K.maximum(K.square(input) - 1, keras.backend.epsilon())))


def get_backward_Arccosh(layer: Arccosh) -> Layer:
    """
    Factory function to create a BackwardArccosh layer from an Arccosh layer.

    ### Parameters:
    - `layer`: A Keras `Arccosh` layer instance.

    ### Returns:
    - A `BackwardArccosh` layer instance.
    """
    return BackwardArccosh(layer)
