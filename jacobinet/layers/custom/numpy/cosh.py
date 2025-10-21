from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Cosh  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardCosh(BackwardNonLinearLayer):
    """
    Implements the backward pass of the Cosh function.

    Given upstream gradient, multiplies by derivative:
    sinh(x)

    Parameters:
    - `layer`: A Keras `Cosh` layer instance.

    Returns:
    - Gradient tensor w.r.t. input.
    """

    def __init__(
        self,
        layer: Cosh,
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
        return gradient * K.sinh(input)


def get_backward_Cosh(layer: Cosh) -> Layer:
    """
    Factory function for BackwardCosh layer from a Cosh layer.
    """
    return BackwardCosh(layer)
