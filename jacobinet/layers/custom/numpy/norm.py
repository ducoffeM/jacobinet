from typing import Any, Optional, Sequence, Union

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy.ops import Norm  # forward layer


@keras.saving.register_keras_serializable()
class BackwardNorm(BackwardNonLinearLayer):
    """
    Implements the backward pass of the Norm layer.

    Given upstream gradient, multiplies by the derivative of the norm
    with respect to the input tensor.

    Derivative formulas:
        For ord = 2: dy/dx = x / ||x||_2
        For ord = p: dy/dx = sign(x) * |x|^{p-1} / ||x||_p^{p-1}

    The gradient must be broadcasted correctly along the reduction axis.
    """

    def __init__(
        self,
        layer: Norm,
        **kwargs: Any,
    ):
        super().__init__(layer=layer, **kwargs)
        self.ord = getattr(layer, "ord", 2)
        self.axis = getattr(layer, "axis", None)

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Compute norm value used in forward
        norm = K.norm(input, ord=self.ord, axis=self.axis, keepdims=True)
        eps = keras.backend.epsilon()

        if self.ord in (None, 2):
            # L2 norm
            grad_input = input / (norm + eps)
        elif self.ord == 1:
            # L1 norm: derivative is sign(x)
            grad_input = K.sign(input)
        else:
            # General p-norm
            abs_x = K.abs(input)
            grad_input = K.sign(input) * (abs_x ** (self.ord - 1)) / (norm ** (self.ord - 1) + eps)

        # Broadcast upstream gradient along reduced axes
        return K.expand_dims(gradient, self.axis) * grad_input


def get_backward_Norm(layer: Norm) -> Layer:
    """
    Factory function for BackwardNorm layer from a Norm layer.
    """
    return BackwardNorm(layer)
