from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import AMin  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardAMin(BackwardNonLinearLayer):
    """
    This function creates a BackwardAMin layer based on a given AMin layer.
    It provides a convenient way to compute the backward pass of the input AMin layer.

    ### Parameters:
    - layer: A Keras AMin layer instance.
      The function uses this layer's configurations to set up the BackwardAMin layer.

    ### Returns:
    - layer_backward: An instance of BackwardAMin, which acts as the reverse layer for the given AMin.

    ### Example Usage:
    ```python
    from keras.layers import AMin
    from keras_custom.backward import get_backward_AMin

    # Assume `amin_layer` is a pre-defined AMin layer
    backward_layer = get_backward_AMin(amin_layer)
    output = backward_layer(input_tensor)
    ```
    """

    def __init__(
        self,
        layer: AMin,
        **kwargs: Any,
    ):
        super().__init__(layer=layer, **kwargs)
        self.axis = getattr(layer, "axis", None)
        self.keepdims = getattr(layer, "keepdims", False)

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if input is None:
            raise ValueError("Input tensor is required for BackwardAMin.")

        # Compute the maximum along the axis
        min_vals = K.amin(input, axis=self.axis, keepdims=self.keepdims)

        if self.keepdims:
            # Create mask where input equals max
            mask = K.cast(K.equal(input, min_vals), K.dtype(input))
        else:
            mask = K.cast(K.equal(input, K.expand_dims(min_vals, self.axis)), K.dtype(input))

        # Handle multiple max values (divide gradient equally)
        num_min = K.sum(mask, axis=self.axis, keepdims=self.keepdims)
        grad_share = gradient / num_min

        # Broadcast and apply mask
        if not self.keepdims:
            grad_share = K.expand_dims(grad_share, axis=self.axis)

        return mask * grad_share


def get_backward_AMin(layer: AMin) -> Layer:
    """
    This function creates a `BackwardAMin` layer based on a given `AMin` layer.
    It provides a convenient way to obtain the backward pass of the input `AMin` layer,
    using the `BackwardAMin`.

    ### Parameters:
    - `layer`: A Keras `AMin` layer instance.
      The function uses this layer's configurations to set up the `BackwardAMin` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardAMin`, which acts as the reverse layer for the given `AMin`.

    ### Example Usage:
    ```python
    from keras.layers import AMin
    from keras_custom.backward import get_backward_AMin

    # Assume `amin_layer` is a pre-defined AMin layer
    backward_layer = get_backward_AMin(amin_layer)
    output = backward_layer(input_tensor)
    ```
    """
    return BackwardAMin(layer)
