from typing import Any, Callable, List, Optional, Tuple, Union

import keras
import keras.ops as K  # type:ignore
import numpy as np
from jacobinet.layers.convert import get_backward as get_backward_layer
from jacobinet.layers.layer import BackwardLayer, BackwardLinearLayer
from keras import KerasTensor as Tensor
from keras.layers import InputLayer  # type:ignore
from keras.layers import Layer  # type:ignore
from keras.losses import Loss  # type:ignore
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
        return (None,) + self.grad_const.shape[1:]

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


def get_model_with_loss(
    model: Union[Model, Sequential], loss: Union[str, Loss, Layer], **kwargs
) -> Tuple[Model, List[Tensor]]:
    """
    Creates a new model that computes the loss based on the given model's output and the provided ground truth.

    This function takes a Keras model and adds a loss computation step to the model by incorporating
    the ground truth input and the given loss function. It returns a new model where the output is the
    computed loss.

    Args:
        model: A Keras model (either a `Model` or `Sequential` instance) that generates predictions.
        loss: The loss function to be used. It can be:
            - A string representing a built-in loss function (e.g., 'categorical_crossentropy').
            - A `Loss` object (Keras loss class).
            - A `Layer` object that computes the loss.
        **kwargs: Additional keyword arguments, including:
            - 'gt_shape': Optional, the shape of the ground truth input. If not provided, the shape is inferred from the model's output.

    Returns:
            - A Keras `Model` that takes the original model's inputs and ground truth as inputs, and outputs the computed loss.
            - A list containing the ground truth input tensor.

    Raises:
        TypeError: If the type of the provided `loss` is not supported (i.e., not a string, `Loss` object, or `Layer`).
        AssertionError: If the output shape of the loss is incorrect (must be a scalar, i.e., shape (None, 1)).

    Example:
        model, _ = get_model_with_loss(my_model, 'categorical_crossentropy')
        model.compile(optimizer='adam', loss='categorical_crossentropy')
    """
    # duplicate inputs of the model

    inputs = [Input(input_i.shape[1:]) for input_i in to_list(model.inputs)]

    # groundtruth target: same shape as model.output if gt_shape undefined in kwargs
    if "gt_shape" in kwargs:
        gt_shape = kwargs["gt_shape"]
    else:
        gt_shape = to_list(model.outputs)[0].shape[1:]

    gt_input = Input(gt_shape)

    loss_layer: Layer
    if isinstance(loss, str):
        loss: Loss = deserialize(loss)
        # convert loss which is a Loss object into a keras Layer
        loss_layer = get_loss_as_layer(loss)
    elif isinstance(loss, Loss):
        # convert loss which is a Loss object into a keras Layer
        loss_layer = get_loss_as_layer(loss)
    elif isinstance(loss, Layer):
        loss_layer = loss
    else:
        raise TypeError("unknown type for loss {}".format(loss.__class__))

    # build a model: loss_layer takes as input y_true, y_pred
    output_pred = model(inputs)
    output_loss = loss_layer([gt_input] + to_list(output_pred))  # (None, 1)
    output_loss_dim_wo_batch = output_loss.shape[1:]
    assert (
        len(output_loss_dim_wo_batch) == 1 and output_loss_dim_wo_batch[0] == 1
    ), "Wrong output shape for model that predicts a loss, Expected [1] got {}".format(
        output_loss_dim_wo_batch
    )

    model_with_loss: Model = keras.models.Model(inputs + [gt_input], output_loss)

    return model_with_loss, [gt_input]


def get_backward_model_with_loss(
    model,
    loss: Union[str, Layer] = "categorical_crossentropy",
    mapping_keras2backward_classes={},
    mapping_keras2backward_losses={},
    **kwargs,
) -> Model:  # we do not compute gradient on extra_inputs, loss should return (None, 1)
    """
    Constructs a backward model that includes a loss computation.
    The model is wrapped such that the loss becomes part
    of the model graph, allowing backpropagation through it.

    Args:
        model: A Keras model (either a `Model` or `Sequential` instance) to be used for generating adversarial examples.
        loss: The loss function used in the model. It can be a string (e.g., 'categorical_crossentropy'), a `Layer` object,
              or a Keras `Loss` object. Defaults to 'categorical_crossentropy'.
        mapping_keras2backward_classes: A dictionary mapping Keras layers to their backward counterparts for gradient computation.
        mapping_keras2backward_losses: A dictionary mapping loss functions to their backward counterparts.
        **kwargs: Additional arguments passed to the `get_model_with_loss` function, such as `gt_shape` or other layer-specific settings.

    Returns:
        Model: A Keras `Model` that computes the backward over the composition of model and loss during training or evaluation. This model includes
               the attack method and loss function, but does not compute gradients for extra inputs.

    Raises:
        NotImplementedError: If the model has multiple inputs or outputs, as this function currently supports single input-output models.

    Example:
        backward_base_model = get_backward_model_with_loss(model=my_model, loss='categorical_crossentropy')
        backward_base_model.compile(optimizer='adam', loss='categorical_crossentropy')
    """

    if len(model.outputs) > 1:
        raise NotImplementedError(
            "actually not working wih multiple loss. Raise a dedicated PR if needed"
        )
    if len(model.inputs) > 1:
        raise NotImplementedError(
            "actually not working wih multiple inputs. Raise a dedicated PR if needed"
        )

    model_with_loss: Model
    label_tensors: List[Tensor]
    model_with_loss, label_tensors = get_model_with_loss(
        model, loss, **kwargs
    )  # to define, same for every atacks

    input_mask = [label_tensor_i.name for label_tensor_i in label_tensors]

    if mapping_keras2backward_classes is None:
        mapping_keras2backward_classes = mapping_keras2backward_losses
    elif not (mapping_keras2backward_losses is None):
        mapping_keras2backward_classes.update(mapping_keras2backward_losses)

    backward_model = clone_to_backward(
        model=model_with_loss,
        mapping_keras2backward_classes=mapping_keras2backward_classes,
        gradient=keras.Variable(np.ones((1, 1), dtype="float32")),
        input_mask=input_mask,
    )
    return backward_model
