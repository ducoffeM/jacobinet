from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Prod  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardProd(BackwardNonLinearLayer):
    """
    Backward layer for the `Prod` operation (product over axes).
    Computes the gradient of a product operation with respect to its input.
    """

    def __init__(
        self,
        layer: Prod,
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
            raise ValueError("Input tensor is required for BackwardProd.")

        # Compute the product along the axis
        prod_vals = K.prod(input, axis=self.axis, keepdims=True)

        # To avoid division by zero, add small epsilon where input is zero
        safe_input = K.where(input == 0, keras.backend.epsilon(), input)

        # Compute the partial derivative: d(prod)/dx_i = prod / x_i
        grad_input = prod_vals / safe_input

        # Reshape the incoming gradient if keepdims is False
        if not self.keepdims:
            gradient = K.expand_dims(gradient, axis=self.axis)

        # Multiply incoming gradient by local gradient
        return gradient * grad_input


def get_backward_Prod(layer: Prod) -> Layer:
    """
    Creates a `BackwardProd` layer for the given `Prod` layer.

    ### Parameters:
    - `layer`: An instance of the `Prod` layer.

    ### Returns:
    - An instance of `BackwardProd`, representing the gradient layer.
    """
    return BackwardProd(layer)
