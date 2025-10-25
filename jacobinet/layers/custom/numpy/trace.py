from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Trace  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardTrace(BackwardLinearLayer):
    use_W_numerically: bool = True
    """
    Implements the backward pass for keras_custom.layers.numpy.Trace.

    ### Forward:
        y = ops.trace(x, offset, axis1, axis2)
    ### Backward:
        dL/dx = diag_embed(dL/dy, offset, axis1, axis2)
    """


@keras.saving.register_keras_serializable()
class BackwardTrace_(BackwardLinearLayer):
    """
    Implements the backward pass for keras_custom.layers.numpy.Trace.

    ### Forward:
        y = ops.trace(x, offset, axis1, axis2)
    ### Backward:
        dL/dx = diag_embed(dL/dy, offset, axis1, axis2)
    """

    def __init__(self, layer: Trace, **kwargs: Any):
        super().__init__(layer=layer, **kwargs)
        self.offset = getattr(layer, "offset", 0)
        self.axis1 = getattr(layer, "axis1", -2)
        self.axis2 = getattr(layer, "axis2", -1)

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if input is None:
            raise ValueError("Input tensor is required for BackwardTrace.")
        if gradient is None:
            raise ValueError("Gradient tensor is required for BackwardTrace.")

        # Prepare a zero tensor of the same shape as input
        grad_input = K.zeros_like(input)

        # Get input shape and axis info
        ndim_in = K.ndim(input)
        axis1 = self.axis1 if self.axis1 >= 0 else ndim_in + self.axis1
        axis2 = self.axis2 if self.axis2 >= 0 else ndim_in + self.axis2
        input_shape = K.shape(input)
        n1 = input_shape[axis1]
        n2 = input_shape[axis2]

        # Compute length of the traced diagonal
        if self.offset >= 0:
            diag_len = K.maximum(K.minimum(n1, n2 - self.offset), 0)
        else:
            diag_len = K.maximum(K.minimum(n1 + self.offset, n2), 0)

        # Create a mask for the diagonal positions with the given offset
        # The mask is shape (n1, n2) with ones on the selected diagonal
        full_eye = K.eye(n1, n2, k=self.offset, dtype=K.dtype(input))

        # Multiply the gradient by the mask to place it along the diagonal
        # Gradient shape may be batched, so expand it to broadcast
        grad_expanded = gradient[..., None, None] * full_eye  # (..., n1, n2)

        # Move the new dims into the correct axes
        grad_expanded = K.moveaxis(grad_expanded, -2, axis1)
        grad_expanded = K.moveaxis(grad_expanded, -1, axis2)

        grad_input = grad_input + grad_expanded
        return grad_input


def get_backward_Trace(layer: Trace) -> Layer:
    """
    Creates a BackwardTrace layer corresponding to a given Trace layer.

    ### Example:
    ```python
    from keras_custom.layers.numpy import Trace
    from keras_custom.backward import get_backward_Trace

    trace_layer = Trace(offset=0, axis1=1, axis2=2)
    backward_layer = get_backward_Trace(trace_layer)

    grad_input = backward_layer(gradient_tensor)
    ```
    """
    return BackwardTrace(layer)
