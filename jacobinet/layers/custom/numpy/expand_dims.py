from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import ExpandDims  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardExpandDims(BackwardLinearLayer):
    """
    Implements the backward pass for keras.ops.expand_dims.

    ### Forward:
        y = expand_dims(x, axis)
    ### Backward:
        dL/dx = sum(dL/dy, axis=axis)

    The backward pass collapses the added dimension by summing gradients.
    """

    def __init__(self, layer: ExpandDims, **kwargs: Any):
        super().__init__(layer=layer, **kwargs)
        self.axis = getattr(layer, "axis", None)
        if self.axis is None:
            raise ValueError("ExpandDims layer must have an 'axis' attribute.")

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if gradient is None:
            raise ValueError("Gradient tensor is required for BackwardExpandDims.")

        # Just remove the dimension by summing over the expanded axis
        grad_input = K.sum(gradient, axis=self.axis, keepdims=False)
        return grad_input


def get_backward_ExpandDims(layer: ExpandDims) -> Layer:
    """
    Creates a BackwardExpandDims layer corresponding to a given ExpandDims layer.

    ### Example:
    ```python
    from keras_custom.layers.numpy import ExpandDims
    from keras_custom.backward import get_backward_ExpandDims

    expand_layer = ExpandDims(axis=1)
    backward_layer = get_backward_ExpandDims(expand_layer)

    output = backward_layer(gradient_tensor)
    ```
    """
    return BackwardExpandDims(layer)
