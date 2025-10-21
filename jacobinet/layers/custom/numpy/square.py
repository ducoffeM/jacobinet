from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Square  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardSquare(BackwardNonLinearLayer):
    """
    This function creates a BackwardSquare layer based on a given Square layer.
    It provides a convenient way to obtain the backward pass of the input Square layer,
    using the BackwardSquare.

    ### Parameters:
    - layer: A Keras Square layer instance.
      The function uses this layer's configurations to set up the BackwardSquare layer.

    ### Returns:
    - layer_backward: An instance of BackwardSquare, which acts as the reverse layer for the given Square.

    ### Example Usage:
    ```python
    from keras.layers import Square
    from keras_custom.backward import get_backward_Square

    # Assume `square_layer` is a pre-defined Square layer
    backward_layer = get_backward_Square(square_layer)
    output = backward_layer(input_tensor)
    ```
    """

    def __init__(
        self,
        layer: Square,
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
        # gradient of square(x) = 2 * x
        return gradient * 2 * input


def get_backward_Square(layer: Square) -> Layer:
    """
    This function creates a `BackwardSquare` layer based on a given `Square` layer.
    It provides a convenient way to obtain the backward pass of the input `Square` layer,
    using the `BackwardSquare`.

    ### Parameters:
    - `layer`: A Keras `Square` layer instance.
      The function uses this layer's configurations to set up the `BackwardSquare` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardSquare`, which acts as the reverse layer for the given `Square`.

    ### Example Usage:
    ```python
    from keras.layers import Square
    from keras_custom.backward import get_backward_Square

    # Assume `square_layer` is a pre-defined Square layer
    backward_layer = get_backward_Square(square_layer)
    output = backward_layer(input_tensor)
    ```
    """
    return BackwardSquare(layer)
