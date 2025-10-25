from typing import Any, Optional, Sequence, Union

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy.ops import Var  # forward layer


@keras.saving.register_keras_serializable()
class BackwardVar(BackwardNonLinearLayer):
    """
    Implements the backward pass of the Var layer.

    Given upstream gradient, computes gradient w.r.t. input:

        dVar/dx = 2 * (x - mean(x)) / (N - ddof)

    and multiplies by the upstream gradient, properly broadcasting
    along the reduced axis.

    Parameters:
    - `layer`: The forward Var layer instance.
    """

    def __init__(
        self,
        layer: Var,
        **kwargs: Any,
    ):
        super().__init__(layer=layer, **kwargs)
        self.axis = getattr(layer, "axis", None)
        self.keepdims = getattr(layer, "keepdims", False)
        self.ddof = getattr(layer, "ddof", 0)

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        axis = self.axis

        # Mean along reduction axes
        mean = K.mean(input, axis=axis, keepdims=True)

        # Number of elements along the reduced axis
        if axis is None:
            N = K.cast(K.size(input), K.floatx())
        else:
            # Handle tuple or int
            if isinstance(axis, int):
                N = K.cast(K.shape(input)[axis], keras.config.floatx())
            else:
                N = K.cast(
                    K.prod(K.stack([K.shape(input)[a] for a in axis])), keras.config.floatx()
                )

        coeff = 2.0 / (N - self.ddof)

        grad_input = coeff * (input - mean)
        return K.expand_dims(gradient, self.axis) * grad_input  # broadcast over reduced axes


def get_backward_Var(layer: Var) -> Layer:
    """
    Factory function for BackwardVar layer from a Var layer.
    """
    return BackwardVar(layer)
