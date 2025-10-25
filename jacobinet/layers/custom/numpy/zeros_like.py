from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import ZerosLike  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardZerosLike(BackwardLinearLayer):
    """
    Backward layer for the `ZerosLike` operation.
    Since the output is constant, the gradient is always zero.
    """

    def __init__(
        self,
        layer: ZerosLike,
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
        # The gradient of a constant function is zero
        return K.zeros_like(gradient)


def get_backward_ZerosLike(layer: ZerosLike) -> Layer:
    """
    Creates a `BackwardZerosLike` layer for the given `ZerosLike` layer.

    ### Parameters:
    - `layer`: An instance of the `ZerosLike` layer.

    ### Returns:
    - An instance of `BackwardZerosLike`, which returns zero gradient.
    """
    return BackwardZerosLike(layer)
