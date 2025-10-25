from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import OnesLike  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardOnesLike(BackwardLinearLayer):
    """
    Backward layer for the `OnesLike` operation.
    Since the output does not depend on the input, the gradient is always zero.
    """

    def __init__(
        self,
        layer: OnesLike,
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
        # Gradient of a constant function is zero
        return K.zeros_like(gradient)


def get_backward_OnesLike(layer: OnesLike) -> Layer:
    """
    Creates a `BackwardOnesLike` layer for the given `OnesLike` layer.

    ### Parameters:
    - `layer`: An instance of the `OnesLike` layer.

    ### Returns:
    - An instance of `BackwardOnesLike`, which returns zero gradient.
    """
    return BackwardOnesLike(layer)
