from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import FullLike  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardFullLike(BackwardLinearLayer):
    """
    Implements the backward pass for keras_custom.layers.numpy.FullLike.

    ### Forward:
        y = keras.ops.full_like(x, fill_value)
        (output has same shape as input, all entries = fill_value)

    ### Backward:
        dL/dx = 0
        Because the output does not depend on input values (only on shape),
        the gradient with respect to the input is zero everywhere.
    """

    def __init__(
        self,
        layer: FullLike,
        **kwargs: Any,
    ):
        super().__init__(layer=layer, **kwargs)

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Gradient of sign is 0 almost everywhere
        return K.zeros_like(gradient)


def get_backward_FullLike(layer: FullLike) -> Layer:
    """
    Creates a BackwardFullLike layer corresponding to a given FullLike layer.

    ### Example
    ```python
    from keras_custom.layers.numpy import FullLike
    from keras_custom.backward import get_backward_FullLike

    full_like_layer = FullLike(fill_value=1.0)
    backward_layer = get_backward_FullLike(full_like_layer)

    grad_input = backward_layer(gradient_tensor)
    ```
    """
    return BackwardFullLike(layer)
