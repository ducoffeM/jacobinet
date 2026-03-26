from typing import Any, Callable, List, Optional, Tuple, Union

import keras
import keras.ops as K  # type:ignore
import numpy as np
from jacobinet.layers.convert import get_backward as get_backward_layer
from jacobinet.layers.layer import BackwardLayer, BackwardLinearLayer
from keras import KerasTensor as Tensor
from keras.layers import InputLayer  # type:ignore
from keras.layers import Layer, Input  # type:ignore
from keras.models import Model, Sequential  # type:ignore


def to_list(tensor: Union[Tensor, List[Tensor]]) -> List[Tensor]:
    """
    Converts a single tensor or a list of tensors to a list of tensors.

    If the input is already a list of tensors, it returns the list unchanged.
    If the input is a single tensor, it wraps it in a list.

    Args:
        tensor: A single tensor or a list of tensors.

    Returns:
        List[Tensor]: A list containing the input tensor(s).

    Example:
        # Single tensor
        tensor = tf.constant([1, 2, 3])
        tensor_list = to_list(tensor)
        print(tensor_list)  # Output: [tensor]

        # List of tensors
        tensor_list = to_list([tensor, tensor])
        print(tensor_list)  # Output: [tensor, tensor]
    """
    if isinstance(tensor, list):
        return tensor
    return [tensor]


def is_linear(model_backward: keras.models.Model) -> bool:
    """
    Checks if a given Keras model is a linear model.

    A model is considered linear if it has an attribute `is_linear` set to `True`.

    Args:
        model_backward: The Keras model to check.

    Returns:
        True if the model is linear, False otherwise.

    Example:
        model = SomeKerasModel()  # Assuming this is a model with the attribute `is_linear`
        print(is_linear(model))  # Output: True or False depending on the model's attributes
    """
    return hasattr(model_backward, "is_linear") and model_backward.is_linear


def is_linear_layer(layer):
    """
    Determines if a given layer is considered a linear layer.

    A layer is considered linear if:
    - It is an instance of `BackwardLinearLayer`, or
    - It has an attribute `is_linear` set to `True`, or
    - It is a layer that is not a `BackwardLayer` and does not explicitly define the `is_linear` attribute (in which case it is treated as linear).

    Args:
        layer: The layer to check.

    Returns:
        `True` if the layer is linear, `False` otherwise.

    Example:
        layer = SomeKerasLayer()  # Assume this layer is linear
        print(is_linear_layer(layer))  # Output: True or False based on the layer's properties
    """
    if not (isinstance(layer, BackwardLayer) or (hasattr(layer, "is_linear"))):
        return True
    return isinstance(layer, BackwardLinearLayer) or (
        hasattr(layer, "is_linear") and layer.is_linear
    )


def get_backward(
    layer: Union[Layer, Model],
    gradient_shape=Union[None, Tuple[int], List[Tuple[int]]],
    mapping_keras2backward_classes: Optional[dict[type[Layer], type[BackwardLayer]]] = None,
    get_backward_layer: Callable = None,
):
    """
    Retrieves the backward computation for a given layer or model.

    This function handles the backward pass for a Keras layer or model. It identifies the type of the input `layer`
    and calls the appropriate function to compute the backward operation for that layer. It is designed to extend
    backward functionality to custom layers or models, leveraging the provided mapping to link Keras layers to their
    backward equivalents.

    Args:
        layer: The Keras layer or model for which to compute the backward pass.
        gradient_shape: The shape of the gradients.
            This is used in certain cases to specify the expected shape of the gradients. Defaults to None.
        mapping_keras2backward_classes: A mapping from Keras layer types
            to their corresponding `BackwardLayer` types. This helps identify the backward operation for custom layers.
            Defaults to None.
        get_backward_layer: A callable that retrieves the backward layer. Defaults to None.

    Returns:
        BackwardLayer or BackwardModel: A backward equivalent of the input layer or model.

    Raises:
        NotImplementedError: If the layer type is not supported and no backward model function is provided.

    Example:
        layer = SomeKerasLayer()
        backward_layer = get_backward(layer)  # Retrieves the corresponding backward layer
    """
    if isinstance(layer, Layer):
        return get_backward_layer(layer, mapping_keras2backward_classes)
    else:
        raise NotImplementedError()


@keras.saving.register_keras_serializable()
class GradConstant(Layer):
    """
    A custom Keras layer that outputs a constant gradient.

    This layer is intended to be used in scenarios where a fixed gradient
    needs to be applied during the backward pass. The provided `gradient`
    is stored as a constant tensor and returned during the backward computation.

    Attributes:
        grad_const (Tensor): The constant gradient tensor to be returned.
    """

    def __init__(self, gradient, input_dim_wo_batch: Union[None, List[int]] = None, **kwargs):
        super(GradConstant, self).__init__(**kwargs)
        self.grad_const = keras.ops.convert_to_tensor(gradient)
        if not input_dim_wo_batch is None:
            self.input_dim_wo_batch = input_dim_wo_batch
        self.output_dim_wo_batch = list(self.grad_const.shape)

    def build(self, input_shape):
        self.input_dim_wo_batch = list(input_shape[1:])

    def call(self, inputs_):
        if isinstance(inputs_, list):
            inputs_ = inputs_[0]
        input_dim_wo_batch = inputs_.shape[1:]

        # avoid disconnected graph
        x = K.reshape(0.0 * inputs_, (-1, np.prod(input_dim_wo_batch)))
        x = K.sum(x, -1)
        grad_const_dim_wo_batch = len(self.grad_const.shape[1:])
        y = K.reshape(x, [-1] + [1] * grad_const_dim_wo_batch)
        return y + self.grad_const

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "grad_const": keras.saving.serialize_keras_object(self.grad_const),
                "input_dim_wo_batch": self.input_dim_wo_batch,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        # Explicitly get the batch size from the input shape (this is often None)

        # Get the shape of your constant, excluding its first dimension
        const_shape = [i for i in self.grad_const.shape[1:]]

        # Combine them using standard tuples to ensure type compatibility
        return (None,) + tuple(const_shape)

    @classmethod
    def from_config(cls, config):
        sublayer_config = config.pop("grad_const")
        sublayer = keras.saving.deserialize_keras_object(sublayer_config)
        return cls(sublayer, **config)


def get_gradient(grad: Tensor, input) -> Tuple[Any, bool]:
    """
    Given a gradient tensor, this function checks if the gradient is an InputLayer
    or KerasTensor. If it is, it returns the gradient as is. Otherwise, it creates a
    constant gradient using the `GradConstant` layer and returns it.

    Args:
        grad: The gradient tensor, which could either be an InputLayer, KerasTensor, or a custom gradient.
        input: The input tensor, passed to `GradConstant` if grad is not a KerasTensor or InputLayer.

    Returns:
            - The gradient or constant gradient layer (which is a `KerasTensor`).
            - A boolean indicating whether the gradient is directly an input (True if it's an input, False otherwise).
    """
    # if grad is an InputLayer return grad
    if isinstance(grad, InputLayer) or isinstance(grad, keras.KerasTensor):
        return (
            grad  # grad is a KerasTensor that come from input or extra_inputs or is an Input Tensor
        )
    # else return it as a Constant of a layer
    constant = GradConstant(gradient=grad)(input)
    return constant


@keras.saving.register_keras_serializable()
class FuseGradients(Layer):
    """
    A custom Keras layer that takes a list of input tensors, expands each tensor
    along a new axis, concatenates them along the last axis, and then sums
    the resulting tensor along the same axis.

    This layer is useful for combining gradients or other tensors into a single
    tensor by first expanding them, stacking them side by side, and then
    summing them together.
    """

    def call(self, inputs, training=None, mask=None):
        # expand
        output = [K.expand_dims(input_, -1) for input_ in inputs]
        # concat
        output = K.concatenate(output, axis=-1)
        # sum
        return K.sum(output, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

