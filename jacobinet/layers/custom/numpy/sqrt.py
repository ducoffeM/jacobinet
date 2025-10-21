from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Sqrt  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardSqrt(BackwardNonLinearLayer):
    """
    This function creates a BackwardSqrt layer based on a given Sqrt layer.
    It provides a convenient way to obtain the backward pass of the input Sqrt layer,
    using the BackwardSqrt.

    ### Parameters:
    - layer: A Keras Sqrt layer instance.
      The function uses this layer's configurations to set up the BackwardSqrt layer.

    ### Returns:
    - layer_backward: An instance of BackwardSqrt, which acts as the reverse layer for the given Sqrt.

    ### Example Usage:
    ```python
    from keras.layers import Sqrt
    from keras_custom.backward import get_backward_Sqrt

    # Assume `sqrt_layer` is a pre-defined Sqrt layer
    backward_layer = get_backward_Sqrt(sqrt_layer)
    output = backward_layer(input_tensor)
    ```
    """

    def __init__(
        self,
        layer: Sqrt,
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
        # gradient of sqrt(x) = 1 / (2 * sqrt(x)) = 1 / (2 * output)
        return gradient / (2 * K.sqrt(input))


def get_backward_Sqrt(layer: Sqrt) -> Layer:
    """
    This function creates a `BackwardSqrt` layer based on a given `Sqrt` layer.
    It provides a convenient way to obtain the backward pass of the input `Sqrt` layer,
    using the `BackwardSqrt`.

    ### Parameters:
    - `layer`: A Keras `Sqrt` layer instance.
      The function uses this layer's configurations to set up the `BackwardSqrt` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardSqrt`, which acts as the reverse layer for the given `Sqrt`.

    ### Example Usage:
    ```python
    from keras.layers import Sqrt
    from keras_custom.backward import get_backward_Sqrt

    # Assume `sqrt_layer` is a pre-defined Sqrt layer
    backward_layer = get_backward_Sqrt(sqrt_layer)
    output = backward_layer(input_tensor)
    ```
    """
    return BackwardSqrt(layer)
