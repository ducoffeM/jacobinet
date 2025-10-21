from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Cos  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardCos(BackwardNonLinearLayer):
    """
    Implements the backward pass of the Cos function.

    Given upstream gradient, multiplies by derivative:
    -sin(x)

    Parameters:
    - `layer`: A Keras `Cos` layer instance.

    Returns:
    - Gradient tensor w.r.t. input.
    """

    def __init__(
        self,
        layer: Cos,
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
        return gradient * (-K.sin(input))


def get_backward_Cos(layer: Cos) -> Layer:
    """
    Factory function for BackwardCos layer from a Cos layer.
    """
    return BackwardCos(layer)
