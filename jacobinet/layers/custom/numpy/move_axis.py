from typing import Any, Optional, Sequence

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import MoveAxis  # forward layer


@keras.saving.register_keras_serializable()
class BackwardMoveAxis(BackwardLinearLayer):
    """
    Implements the backward pass of the MoveAxis layer.

    Given upstream gradient, re‐permutes axes to match original input gradient shape.

    The forward MoveAxis takes arguments `source` and `destination`.
    Here we invert by swapping source and destination.
    """

    def __init__(
        self,
        layer: MoveAxis,
        **kwargs: Any,
    ):
        super().__init__(layer=layer, **kwargs)
        # capture source & destination for backward
        self.source: Sequence[int] = layer.source
        self.destination: Sequence[int] = layer.destination

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # invert the permutation
        return K.moveaxis(gradient, self.destination, self.source)


def get_backward_MoveAxis(layer: MoveAxis) -> Layer:
    return BackwardMoveAxis(layer)
