from typing import Any, Optional, Sequence

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Transpose  # forward layer


@keras.saving.register_keras_serializable()
class BackwardTranspose(BackwardLinearLayer):
    """
    Implements the backward pass of the Transpose layer.

    Given upstream gradient, applies the inverse permutation of axes
    to get the gradient w.r.t. the input.

    Since transpose is a linear axis permutation, its backward is just
    another transpose with the inverse permutation.
    """

    def __init__(
        self,
        layer: Transpose,
        **kwargs: Any,
    ):
        super().__init__(layer=layer, **kwargs)
        self.axes = getattr(layer, "axes", None)

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        axes = self.axes
        if axes is None:
            # Default transpose reverses axes, same for backward
            return K.transpose(gradient)
        else:
            # Compute inverse permutation
            inv_axes = [0] * len(axes)
            for i, a in enumerate(axes):
                inv_axes[a] = i
            return K.transpose(gradient, axes=inv_axes)


def get_backward_Transpose(layer: Transpose) -> Layer:
    """
    Factory function for BackwardTranspose layer from a Transpose layer.
    """
    return BackwardTranspose(layer)
