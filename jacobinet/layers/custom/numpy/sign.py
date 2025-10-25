from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Sign  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardSign(BackwardLinearLayer):
    """
    Backward layer for the `Sign` operation.
    The derivative of sign(x) is zero almost everywhere.
    """

    def __init__(
        self,
        layer: Sign,
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
        # Gradient of sign is 0 almost everywhere
        return K.zeros_like(gradient)


def get_backward_Sign(layer: Sign) -> Layer:
    """
    Creates a `BackwardSign` layer for the given `Sign` layer.

    ### Parameters:
    - `layer`: An instance of the `Sign` layer.

    ### Returns:
    - An instance of `BackwardSign`, which outputs zero gradients.
    """
    return BackwardSign(layer)
