from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import SwapAxes  # forward layer


@keras.saving.register_keras_serializable()
class BackwardSwapAxes(BackwardLinearLayer):
    """
    Implements the backward pass of the SwapAxes layer.

    Given upstream gradient, swaps the same two axes to obtain
    the gradient w.r.t. input.

    The backward pass of swapaxes is identical to the forward pass,
    since swapping twice restores the original axis order.
    """

    def __init__(
        self,
        layer: SwapAxes,
        **kwargs: Any,
    ):
        super().__init__(layer=layer, **kwargs)
        self.layer = layer
        self.axis1 = layer.axis1
        self.axis2 = layer.axis2

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Swapping same axes undoes the forward swap
        return K.swapaxes(gradient, self.axis1, self.axis2)


def get_backward_SwapAxes(layer: SwapAxes) -> Layer:
    """
    Factory function for BackwardSwapAxes layer from a SwapAxes layer.
    """
    return BackwardSwapAxes(layer)
