from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Sort  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardSort(BackwardNonLinearLayer):
    """
    Backward layer for the `Sort` operation.
    The gradient is 'unsorted' to match the original input order.
    """

    def __init__(
        self,
        layer: Sort,
        **kwargs: Any,
    ):
        super().__init__(layer=layer, **kwargs)
        self.axis = getattr(layer, "axis", -1)

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if input is None:
            raise ValueError("Input tensor is required for BackwardSort.")

        input_sorted_indices = K.argsort(input, axis=self.axis)

        # Compute the inverse permutation
        inverse_indices = K.argsort(input_sorted_indices, axis=self.axis)

        # Unsort the gradient
        grad_input = K.take_along_axis(gradient, indices=inverse_indices, axis=self.axis)
        return grad_input


def get_backward_Sort(layer: Sort) -> Layer:
    """
    Creates a `BackwardSort` layer for the given `Sort` layer.

    ### Parameters:
    - `layer`: An instance of the `Sort` layer.

    ### Returns:
    - An instance of `BackwardSort`, which reverses the sort permutation during backprop.
    """
    return BackwardSort(layer)
