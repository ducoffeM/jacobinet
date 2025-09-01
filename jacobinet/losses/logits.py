import keras
import keras.ops as K  # type:ignore

from .base_loss import BackwardLoss
from .loss import Logits_Layer


@keras.saving.register_keras_serializable()
class BackwardLogits(BackwardLoss):
    """
    A custom backward loss layer for computing gradients in logits loss functions,
    specifically for logits without a softmax activation.

    This class is designed to compute the backward gradients for the logits
    loss function, where the forward pass typically does not involve softmax. The gradients are computed
    with respect to both the true labels (`y_true`) and the predicted logits (`y_pred`).

    Args:
        layer: The layer that provides the loss function used for
                                                logits and its associated parameters.

    Raises:
        NotImplementedError: If the `loss.from_logits` is `False` and the softmax as an activation
                              has not been implemented yet.

    Example:
        ```python
        backward_crossentropy = BackwardCrossentropy(layer=my_loss_layer)
        gradients = backward_crossentropy.call_on_reshaped_gradient(gradient, input_data)
        ```
    """

    layer: Logits_Layer

    def call_on_reshaped_gradient(self, gradient, input=None, training=None, mask=None):
        y_true, y_pred = input

        return [gradient * y_pred, gradient * y_true]


def get_backward_Logits(
    layer: Logits_Layer,
) -> BackwardLogits:
    """
    This function creates a `BackwardLogits` layer based on a given `Logits_Layer` layer. It provides
    a convenient way to obtain a backward approximation of the input `Logits_Layer` layer, using the
    `BackwardLogits` class to reverse the flatten operation.

    ### Parameters:
    - `layer`: A Keras `Logits_Layer` layer instance. The function uses this layer's configurations to set up the `BackwardLogits` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardLogits`, which acts as the reverse layer for the given `Logits_Layer`.

    ### Example Usage:
    ```python
    from keras.losses import Logits
    from jacobinet.losses get_loss_as_layer
    from keras_custom.backward import get_backward_Flatten

    # Assume `loss` is a pre-defined Logits loss
    loss_layer = get_loss_as_layer(loss)
    backward_layer = get_backward_Flatten(loss_layer)
    output = backward_layer([y_true, y_pred])
    """

    return BackwardLogits(layer)
