from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Average  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardAverage(BackwardLinearLayer):
    """
    Backward layer for the `Average` operation (mean over an axis).
    Distributes gradient evenly over the inputs along the specified axis.
    """

    def __init__(
        self,
        layer: Average,
        **kwargs: Any,
    ):
        super().__init__(layer=layer, **kwargs)
        self.axis = getattr(layer, "axis", None)
        self.keepdims = getattr(layer, "keepdims", False)

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if input is None:
            raise ValueError("Input tensor is required for BackwardAverage.")

        axis = self.axis
        input_shape = K.shape(input)

        # Calculate number of elements averaged over
        if axis is None:
            # Average over all elements
            count = K.prod(input_shape)
        else:
            # axis can be int or tuple
            if isinstance(axis, int):
                axis_tuple = (axis,)
            else:
                axis_tuple = axis

            # multiply sizes of axes averaged over
            dims = [input_shape[ax] for ax in axis_tuple]
            count = dims[0]
            for d in dims[1:]:
                count = count * d

        # Reshape incoming gradient if keepdims is False
        if not self.keepdims and axis is not None:
            if isinstance(axis, int):
                gradient = K.expand_dims(gradient, axis=axis)
            else:
                # For tuple axis, expand dims in reversed order to preserve shape
                for ax in sorted(axis_tuple):
                    gradient = K.expand_dims(gradient, axis=ax)

        # Distribute gradient evenly
        grad_input = gradient / K.cast(count, K.dtype(gradient))

        # Broadcast to input shape
        grad_input = K.broadcast_to(grad_input, input_shape)

        return grad_input


def get_backward_Average(layer: Average) -> Layer:
    """
    Creates a `BackwardAverage` layer for the given `Average` layer.

    ### Parameters:
    - `layer`: An instance of the `Average` layer.

    ### Returns:
    - An instance of `BackwardAverage`, which distributes the gradient equally.
    """
    return BackwardAverage(layer)
