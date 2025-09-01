from typing import Callable, List, Optional, Union

from jacobinet.layers.convert import get_backward as get_backward_layer
from jacobinet.layers.layer import (
    BackwardLayer,
    BackwardLinearLayer,
    BackwardNonLinearLayer,
)
from keras import KerasTensor as Tensor  # type:ignore
from keras.layers import Input, InputLayer, Layer  # type:ignore
from keras.models import Sequential  # type:ignore

from .base_model import BackwardModel, BackwardSequential
from .utils import get_gradient, is_linear_layer


def get_backward_sequential(
    model: Sequential,
    gradient: Union[None, Tensor, List[Tensor]] = None,
    mapping_keras2backward_classes: Optional[dict[type[Layer], type[BackwardLayer]]] = None,
    extra_inputs: List[Input] = [],
    get_backward: Callable = get_backward_layer,
) -> Union[BackwardSequential, BackwardModel]:
    """
    Computes the backward propagation of gradients through a Sequential Keras model.

    This function constructs a backward model for a given Keras `Sequential` model. It supports both linear and non-linear layers,
    and handles the gradient computation accordingly. If the model is linear, it computes the gradients directly through the layers.
    If the model is non-linear, it adjusts the computation by considering the individual layer's backward operations and chaining them.

    Args:
        model: The Keras Sequential model for which the backward propagation is to be computed.
        gradient: The gradient tensor to be propagated backward through the model.
            If `None`, a placeholder for the gradient is created, and it will be computed automatically.
        mapping_keras2backward_classes: A dictionary that maps Keras layer types to their corresponding
            backward layer classes. This is used to obtain the backward-compatible version of the layers.
        extra_inputs: A list of extra inputs that may need to be added to the backward model.
        get_backward: A callable that returns the backward version of a given Keras layer. Defaults to `get_backward_layer`.

    Returns:
        BackwardModel: The backward model constructed from the input Keras `Sequential` model. This model includes all layers in reverse order,
        and supports backward propagation of gradients.

    Raises:
        NotImplementedError: If certain conditions arise that the function cannot handle, such as non-linear models with unsupported configurations.

    Example:
        ```python
        backward_model = get_backward_sequential(model, gradient)
        ```
    """
    # get input_dim without batch
    input_dim_wo_batch = list(model.inputs[0].shape[1:])
    # for output_tensor in model.outputs:
    output_dim_wo_batch = list(model.outputs[0].shape[1:])

    grad_input = False
    if gradient is not None:
        if isinstance(gradient, List):
            grad = gradient[0]
        else:
            grad = gradient

        if hasattr(grad, "_keras_history"):
            grad_input = isinstance(grad._keras_history.operation, InputLayer)
        else:
            grad_input = False

    # convert every layers
    layers_backward = [
        get_backward(
            layer,
            mapping_keras2backward_classes=mapping_keras2backward_classes,
        )
        for layer in model.layers
    ]
    # check if the layers are all linear
    is_linear = min([is_linear_layer(layer_backward) for layer_backward in layers_backward])
    if is_linear:
        if gradient is None:
            backward_model = BackwardSequential(layers=layers_backward[::-1])
            # init shape
            if len(model.outputs) == 1:
                backward_model(model.outputs[0])  # build correctly the sequential model
            else:
                backward_model(model.outputs)  # Sequential.build() does nothing sadly
            return backward_model
        elif grad_input or len(extra_inputs):
            output = None
            for layer in layers_backward[::-1]:
                if output is None:
                    output = layer(gradient)
                else:
                    output = layer(gradient)
            if grad_input:
                return BackwardModel(output, gradient)
            else:
                return BackwardModel(extra_inputs, output)
        else:
            raise NotImplementedError()

    else:
        input_tensor = Input(input_dim_wo_batch)

        if gradient is None:
            gradient = Input(output_dim_wo_batch)
            grad_input = True
        else:
            gradient = get_gradient(gradient, input_tensor)

        # forward propagation
        dico_input_layer = dict()
        output = None
        for layer, backward_layer in zip(model.layers, layers_backward):
            if output is None:
                if isinstance(backward_layer, BackwardNonLinearLayer):
                    dico_input_layer[id(backward_layer)] = input_tensor
                elif isinstance(backward_layer, BackwardModel) and not backward_layer.is_linear:
                    dico_input_layer[id(backward_layer)] = input_tensor
                output = layer(input_tensor)
            else:
                if isinstance(backward_layer, BackwardNonLinearLayer):
                    dico_input_layer[id(backward_layer)] = output
                elif isinstance(backward_layer, BackwardModel) and not backward_layer.is_linear:
                    dico_input_layer[id(backward_layer)] = output
                output = layer(output)

        output_backward = None
        for backward_layer in layers_backward[::-1]:
            if output_backward is None:
                input_backward = [gradient]
            else:
                input_backward = [output_backward]
            if isinstance(backward_layer, BackwardLinearLayer):
                # no need for forward input
                output_backward = backward_layer(input_backward[0])
            elif hasattr(backward_layer, "is_linear") and backward_layer.is_linear:
                # no need for forward input
                output_backward = backward_layer(input_backward[0])
            elif isinstance(backward_layer, BackwardModel) or isinstance(
                backward_layer, BackwardSequential
            ):
                input_forward = dico_input_layer[id(backward_layer)]
                input_backward = [input_forward] + input_backward
                output_backward = backward_layer(input_backward)
            else:
                input_forward = dico_input_layer[id(backward_layer)]
                # input_backward = [input_forward]+input_backward
                input_backward.append(input_forward)
                output_backward = backward_layer(input_backward)

        inputs = [input_tensor] + extra_inputs
        if grad_input:
            inputs.append(gradient)

        return BackwardModel(inputs, output_backward)
