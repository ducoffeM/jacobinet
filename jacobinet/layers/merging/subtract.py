from typing import Any, List, Optional

import keras
from jacobinet.layers.merging.base_merge import BackwardMergeLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer, Subtract  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardSubtract(BackwardMergeLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Subtract` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Subtract
    from keras_custom.backward.layers import BackwardSubtract

    # Assume `sub_layer` is a pre-defined Subtract layer
    backward_layer = BackwardSubtract(sub_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Subtract,
        **kwargs: Any,
    ):
        super().__init__(layer=layer, **kwargs)

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[List[Tensor]] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> List[Tensor]:
        return [gradient, -gradient]


def get_backward_Subtract(layer: Subtract) -> Layer:
    """
    This function creates a `BackwardSubtract` layer based on a given `Subtract` layer. It provides
    a convenient way to obtain the backward pass of the input `Subtract` layer, using the
    `BackwardSubtract`.

    ### Parameters:
    - `layer`: A Keras `Subtract` layer instance. The function uses this layer's configurations to set up the `BackwardSubtract` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardSubtract`, which acts as the reverse layer for the given `Subtract`.

    ### Example Usage:
    ```python
    from keras.layers import Subtract
    from keras_custom.backward import get_backward_Subtract

    # Assume `sub_layer` is a pre-defined Subtract layer
    backward_layer = get_backward_Subtract(sub_layer)
    output = backward_layer(input_tensor)
    """
    return BackwardSubtract(layer)
