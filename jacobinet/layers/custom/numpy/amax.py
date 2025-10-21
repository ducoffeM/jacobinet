from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Amax  # type:ignore


@keras.saving.register_keras_serializable()
class BackwardAmax(BackwardNonLinearLayer):
    """
    This function creates a `BackwardAmax` layer based on a given `Amax` layer. It provides
    a convenient way to obtain the backward pass of the input `Amax` layer, using the
    `BackwardAmax`.

    ### Parameters:
    - `layer`: A Keras `Amax` layer instance. The function uses this layer's configurations to set up the `BackwardAmax` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardAmax`, which acts as the reverse layer for the given `Amax`.

    ### Example Usage:
    ```python
    from keras.layers import Amax
    from keras_custom.backward import get_backward_Amax

    # Assume `amax_layer` is a pre-defined Amax layer
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Amax,
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
        raise NotImplementedError()


def get_backward_Amax(layer: Amax) -> Layer:
    """
    This function creates a `BackwardAmax` layer based on a given `Amax` layer. It provides
    a convenient way to obtain the backward pass of the input `Amax` layer, using the
    `BackwardAmax`.

    ### Parameters:
    - `layer`: A Keras `Amax` layer instance. The function uses this layer's configurations to set up the `BackwardAmax` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardAmax`, which acts as the reverse layer for the given `Amax`.

    ### Example Usage:
    ```python
    from keras.layers import Amax
    from keras_custom.backward import get_backward_Amax

    # Assume `amax_layer` is a pre-defined Amax layer
    backward_layer = get_backward_Amax(amax_layer)
    output = backward_layer(input_tensor)
    """
    return BackwardAmax(layer)
