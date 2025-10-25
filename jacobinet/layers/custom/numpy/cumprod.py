from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Cumprod  # type: ignore


def reverse_along_axis(x: Tensor, axis: int) -> Tensor:
    """Reverse a tensor along a given axis using slicing."""
    # Ensure positive axis
    ndim = K.ndim(x)
    if axis < 0:
        axis = ndim + axis

    # Build slicing tuple: all slices are [:], except target axis which is reversed
    slices = [slice(None)] * ndim
    slices[axis] = slice(None, None, -1)
    return x[tuple(slices)]


@keras.saving.register_keras_serializable()
class BackwardCumProd(BackwardNonLinearLayer):
    """
    Backward layer for the `CumProd` operation (cumulative product along an axis).
    Computes the gradient of a cumprod operation with respect to its input.
    """

    def __init__(
        self,
        layer: Cumprod,
        **kwargs: Any,
    ):
        super().__init__(layer=layer, **kwargs)
        self.axis = getattr(layer, "axis", None)

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if input is None:
            raise ValueError("Input tensor is required for BackwardCumprod.")

        # Forward cumulative product (y)
        y = K.cumprod(input, axis=self.axis)

        # Avoid division by zero
        safe_input = K.where(K.not_equal(input, 0), input, K.ones_like(input))

        # Compute gradient contributions
        grad_over_x = gradient * (y / safe_input)

        # Reverse cumulative sum along axis
        rev_grad_over_x = K.flip(grad_over_x, self.axis)
        rev_cumsum = K.cumsum(rev_grad_over_x, axis=self.axis)
        cumsum = K.flip(rev_cumsum, self.axis)

        # Compute the reverse cumulative sum along the same axis
        # Need to flip, cumsum, and flip back to simulate "reverse" accumulation
        # rev_grad_over_x = K.reverse(grad_over_x, axis=[self.axis])
        # rev_cumsum = K.cumsum(rev_grad_over_x, axis=self.axis)
        # cumsum = K.reverse(rev_cumsum, axis=[self.axis])

        # Final gradient: dL/dx = cumsum * y / x
        grad_input = cumsum * (y / safe_input)

        return grad_input

    def call_on_reshaped_gradient_(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if input is None:
            raise ValueError("Input tensor is required for BackwardCumProd.")

        axis = self.axis if self.axis is not None else 0

        # Compute the cumulative product of input along axis, with same config
        cumprod_input = K.cumprod(input, axis=axis)

        # To avoid division by zero
        safe_input = K.where(input == 0, keras.backend.epsilon(), input)

        # Compute left and right partial products needed for gradient

        # Left partial product: cumprod of input excluding current element
        # (exclusive cumprod with reverse if needed)
        left = K.cumprod(
            input,
            axis=axis,
        )

        # Right partial product: cumprod of input excluding current element on the right side
        right = K.cumprod(
            input,
            axis=axis,
        )

        # The gradient w.r.t input_i is:
        # grad_input_i = sum_{j>=i} grad_output_j * (prod_{k=i+1}^{j} x_k)
        # But this is equivalent to grad_output_i * left_i * right_i

        # Final gradient = incoming gradient * left partial product * right partial product
        grad_input = gradient * left * right

        return grad_input


def get_backward_CumProd(layer: Cumprod) -> Layer:
    """
    Creates a `BackwardCumProd` layer for the given `CumProd` layer.

    ### Parameters:
    - `layer`: An instance of the `CumProd` layer.

    ### Returns:
    - An instance of `BackwardCumProd`, representing the gradient layer.
    """
    return BackwardCumProd(layer)
