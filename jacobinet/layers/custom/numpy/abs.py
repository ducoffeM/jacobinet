from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Abs  # type:ignore


@keras.saving.register_keras_serializable()
class BackwardAbs(BackwardNonLinearLayer):
    """
    This function creates a `BackwardAbs` layer based on a given `Abs` layer. It provides
    a convenient way to obtain the backward pass of the input `Abs` layer, using the
    `BackwardAbs`.

    ### Parameters:
    - `layer`: A Keras `Abs` layer instance. The function uses this layer's configurations to set up the `BackwardAbs` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardAbs`, which acts as the reverse layer for the given `Abs`.

    ### Example Usage:
    ```python
    from keras.layers import Abs
    from keras_custom.backward import get_backward_Abs

    # Assume `abs_layer` is a pre-defined Abs layer
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Abs,
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
        return gradient * K.sign(input)


def get_backward_Abs(layer: Abs) -> Layer:
    """
    This function creates a `BackwardAbs` layer based on a given `Abs` layer. It provides
    a convenient way to obtain the backward pass of the input `Abs` layer, using the
    `BackwardAbs`.

    ### Parameters:
    - `layer`: A Keras `Abs` layer instance. The function uses this layer's configurations to set up the `BackwardAbs` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardAbs`, which acts as the reverse layer for the given `Abs`.

    ### Example Usage:
    ```python
    from keras.layers import Abs
    from keras_custom.backward import get_backward_Abs

    # Assume `abs_layer` is a pre-defined Abs layer
    backward_layer = get_backward_Abs(abs_layer)
    output = backward_layer(input_tensor)
    """
    return BackwardAbs(layer)


def get_backward_Absolute(layer: Abs) -> Layer:
    """
    This function creates a `BackwardAbs` layer based on a given `Absolute` layer (Abs is a shorthand for Absolute). It provides
    a convenient way to obtain the backward pass of the input `Abs` layer, using the
    `BackwardAbs`.

    ### Parameters:
    - `layer`: A Keras `Absolute` layer instance. The function uses this layer's configurations to set up the `BackwardAbs` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardAbs`, which acts as the reverse layer for the given `Abs`.

    ### Example Usage:
    ```python
    from keras.layers import Abs
    from keras_custom.backward import get_backward_Abs

    # Assume `abs_layer` is a pre-defined Absolute layer
    backward_layer = get_backward_Absolute(abs_layer)
    output = backward_layer(input_tensor)
    """
    return BackwardAbs(layer)
