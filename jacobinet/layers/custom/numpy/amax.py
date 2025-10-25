from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import AMax  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardAMax(BackwardNonLinearLayer):
    """
    This function creates a BackwardAMax layer based on a given AMax layer.
    It provides a convenient way to compute the backward pass of the input AMax layer.

    ### Parameters:
    - layer: A Keras AMax layer instance.
      The function uses this layer's configurations to set up the BackwardAMax layer.

    ### Returns:
    - layer_backward: An instance of BackwardAMax, which acts as the reverse layer for the given AMax.

    ### Example Usage:
    ```python
    from keras.layers import AMax
    from keras_custom.backward import get_backward_AMax

    # Assume `amax_layer` is a pre-defined AMax layer
    backward_layer = get_backward_AMax(amax_layer)
    output = backward_layer(input_tensor)
    ```
    """

    def __init__(
        self,
        layer: AMax,
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
            raise ValueError("Input tensor is required for BackwardAMax.")

        # Compute the maximum along the axis
        max_vals = K.amax(input, axis=self.axis, keepdims=self.keepdims)
        if self.keepdims:
            # Create mask where input equals max
            mask = K.cast(K.equal(input, max_vals), K.dtype(input))
        else:
            mask = K.cast(K.equal(input, K.expand_dims(max_vals, self.axis)), K.dtype(input))

        # Handle multiple max values (divide gradient equally)
        num_max = K.sum(mask, axis=self.axis, keepdims=self.keepdims)
        grad_share = gradient / num_max

        # Broadcast and apply mask
        if not self.keepdims:
            grad_share = K.expand_dims(grad_share, axis=self.axis)
        return mask * grad_share


def get_backward_AMax(layer: AMax) -> Layer:
    """
    This function creates a `BackwardAMax` layer based on a given `AMax` layer.
    It provides a convenient way to obtain the backward pass of the input `AMax` layer,
    using the `BackwardAMax`.

    ### Parameters:
    - `layer`: A Keras `AMax` layer instance.
      The function uses this layer's configurations to set up the `BackwardAMax` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardAMax`, which acts as the reverse layer for the given `AMax`.

    ### Example Usage:
    ```python
    from keras.layers import AMax
    from keras_custom.backward import get_backward_AMax

    # Assume `amax_layer` is a pre-defined AMax layer
    backward_layer = get_backward_AMax(amax_layer)
    output = backward_layer(input_tensor)
    ```
    """
    return BackwardAMax(layer)
