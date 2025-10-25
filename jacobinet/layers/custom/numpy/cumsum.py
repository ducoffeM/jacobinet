from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Cumsum  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardCumsum(BackwardNonLinearLayer):
    """
    Backward pass for keras.ops.cumsum.

    Gradient rule:
        dL/dx = grad_output @ (upper triangular of ones)
    """

    def __init__(self, layer: Cumsum, **kwargs: Any):
        super().__init__(layer=layer, **kwargs)
        self.axis = getattr(layer, "axis", -1)

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if gradient is None:
            raise ValueError("Gradient tensor is required for BackwardCumsum.")

        shape = K.shape(gradient)
        axis_len = shape[self.axis]

        # Create an upper-triangular matrix of 1s (size: axis_len x axis_len)
        tri = K.triu(K.ones((axis_len, axis_len), dtype=K.dtype(gradient)))

        # We need to perform a batched matmul along the chosen axis.
        # Bring the axis to the last position for easier computation.
        perm = list(range(K.ndim(gradient)))
        perm[self.axis], perm[-1] = perm[-1], perm[self.axis]
        grad_perm = K.transpose(gradient, perm)

        # Apply matrix multiplication along the last dimension
        grad_expanded = K.expand_dims(grad_perm, axis=-2)
        grad_out = K.matmul(grad_expanded, tri.T)
        grad_out_ = K.squeeze(grad_out, axis=-2)

        # Undo the transpose to restore original axis order
        grad_input = K.transpose(grad_out_, perm)

        return grad_input


def get_backward_Cumsum(layer: Cumsum) -> Layer:
    """Creates a BackwardCumsum layer for the given Cumsum layer."""
    return BackwardCumsum(layer)
