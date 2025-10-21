from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Tan  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardTan(BackwardNonLinearLayer):
    """
    Implements the backward pass of the Tan function.

    Given upstream gradient, multiplies by derivative:
    1 + tan^2(x)

    Parameters:
    - `layer`: A Keras `Tan` layer instance.

    Returns:
    - Gradient tensor w.r.t. input.
    """

    def __init__(
        self,
        layer: Tan,
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
        return gradient * (1 + K.square(K.tan(input)))


def get_backward_Tan(layer: Tan) -> Layer:
    """
    Factory function for BackwardTan layer from a Tan layer.
    """
    return BackwardTan(layer)
