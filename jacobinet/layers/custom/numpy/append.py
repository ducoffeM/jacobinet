from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Append  # type:ignore


@keras.saving.register_keras_serializable()
class BackwardAppend(BackwardLinearLayer):
    """
    This function creates a `BackwardAppend` layer based on a given `Append` layer. It provides
    a convenient way to obtain the backward pass of the input `Append` layer, using the
    `BackwardAppend`.

    ### Parameters:
    - `layer`: A Keras `Append` layer instance. The function uses this layer's configurations to set up the `BackwardAppend` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardAppend`, which acts as the reverse layer for the given `Append`.

    ### Example Usage:
    ```python
    from keras_custom.layers import Append
    from custom.backward import get_backward_Append

    # Assume `append_layer` is a pre-defined Append layer
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Append,
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


def get_backward_Append(layer: Append) -> Layer:
    """
    This function creates a `BackwardAppend` layer based on a given `Append` layer. It provides
    a convenient way to obtain the backward pass of the input `Append` layer, using the
    `BackwardAppend`.

    ### Parameters:
    - `layer`: A Keras `Append` layer instance. The function uses this layer's configurations to set up the `BackwardAppend` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardAppend`, which acts as the reverse layer for the given `Append`.

    ### Example Usage:
    ```python
    from keras_custom.layers import Append
    from custom.backward import get_backward_Append

    # Assume `append_layer` is a pre-defined Append layer
    backward_layer = get_backward_Append(append_layer)
    output = backward_layer(input_tensor)
    """
    return BackwardAppend(layer)
