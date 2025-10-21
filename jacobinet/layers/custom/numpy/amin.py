from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Amin  # type:ignore


@keras.saving.register_keras_serializable()
class BackwardAmin(BackwardNonLinearLayer):
    """
    This function creates a `BackwardAmin` layer based on a given `Amin` layer. It provides
    a convenient way to obtain the backward pass of the input `Amin` layer, using the
    `BackwardAmin`.

    ### Parameters:
    - `layer`: A Keras `Amin` layer instance. The function uses this layer's configurations to set up the `BackwardAmin` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardAmin`, which acts as the reverse layer for the given `Amin`.

    ### Example Usage:
    ```python
    from keras.layers import Amin
    from keras_custom.backward import get_backward_Amin

    # Assume `amin_layer` is a pre-defined Amin layer
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Amin,
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


def get_backward_Amin(layer: Amin) -> Layer:
    """
    This function creates a `BackwardAmin` layer based on a given `Amin` layer. It provides
    a convenient way to obtain the backward pass of the input `Amin` layer, using the
    `BackwardAmin`.

    ### Parameters:
    - `layer`: A Keras `Amin` layer instance. The function uses this layer's configurations to set up the `BackwardAmin` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardAmin`, which acts as the reverse layer for the given `Amin`.

    ### Example Usage:
    ```python
    from keras.layers import Amin
    from keras_custom.backward import get_backward_Amin

    # Assume `amin_layer` is a pre-defined Amin layer
    backward_layer = get_backward_Amin(amax_layer)
    output = backward_layer(input_tensor)
    """
    return BackwardAmin(layer)
