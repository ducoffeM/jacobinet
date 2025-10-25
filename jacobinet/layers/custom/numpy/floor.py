from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Floor  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardFloor(BackwardLinearLayer):
    """
    Backward layer for the `Floor` operation.
    The gradient of floor is zero almost everywhere.
    """

    def __init__(
        self,
        layer: Floor,
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
        # Gradient of floor function is zero everywhere
        return K.zeros_like(gradient)


def get_backward_Floor(layer: Floor) -> Layer:
    """
    Creates a `BackwardFloor` layer for the given `Floor` layer.

    ### Parameters:
    - `layer`: An instance of the `Floor` layer.

    ### Returns:
    - An instance of `BackwardFloor`, which outputs zero gradients.
    """
    return BackwardFloor(layer)
