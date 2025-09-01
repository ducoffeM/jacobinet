from typing import List, Optional, Union

from jacobinet.layers.convert import get_backward as get_backward_layer
from jacobinet.layers.layer import BackwardLayer
from keras import KerasTensor as Tensor
from keras.layers import Input, InputLayer, Layer  # type:ignore
from keras.models import Model  # type:ignore

from .base_model import BackwardModel
from .node import get_backward_node
from .utils import get_gradient, to_list


def get_backward_functional(
    model: Model,
    gradient: Union[None, Tensor, List[Tensor]] = None,
    mapping_keras2backward_classes: Optional[dict[type[Layer], type[BackwardLayer]]] = None,
    extra_inputs: Union[List[Input]] = [],
    input_mask=None,
    target_inputs=None,
    get_backward: callable = get_backward_layer,
) -> BackwardModel:
    """
    Converts a Keras functional model into a backward-compatible model, supporting gradient-based backward computations.

    This function generates a backward-compatible model by tracing the backward gradients through each layer of the model.
    It supports both functional and sequential Keras models, and maps the output gradients to the corresponding backward layers.

    Args:
        model: The original Keras model to convert to a backward-compatible model.
        gradient: The gradient tensor(s) used for backward computations.
            If `None`, new gradient tensors will be created for each output node.
        mapping_keras2backward_classes: A mapping from Keras layer types to their corresponding
            backward layer classes (e.g., `dict[type[Layer], type[BackwardLayer]]`).
        extra_inputs: Additional inputs to the backward model.
        input_mask: A mask for input layers to apply in backward computation, determining which inputs to include.
        target_inputs: The target input layers for backward computations. Defaults to the model's original inputs.
        get_backward: A function used to get the backward representation of a given layer.
            Defaults to `get_backward_layer`.

    Returns:
        BackwardModel: A backward-compatible version of the original Keras model, with gradients and backward layers.

    Raises:
        ValueError: If there is a mismatch between the gradient and output nodes, or unsupported layer types are encountered.
        NotImplementedError: If certain input or target input masking is not implemented.

    Example:
        ```python
        backward_model = get_backward_functional(model, gradient=some_gradient)
        ```
    """
    # find output_nodes
    grad_input = (
        []
    )  # list of boolean to check whether the gradient should be used as an input of the backward model
    if gradient is not None:
        if not isinstance(gradient, List):
            gradient = [gradient]

        for grad in gradient:
            if grad is None:
                raise ValueError(
                    "None values among not None values of gradient is not considered. Mismatch between gradient and output nodes: The gradient must be specified for every output node, or not specified at all."
                )
            if hasattr(grad, "_keras_history"):
                grad_input_i = isinstance(grad._keras_history.operation, InputLayer)
            else:
                grad_input_i = False

            grad_input.append(grad_input_i)

    # convert model inputs and outputs as list
    model_inputs = to_list(model.input)
    if target_inputs is None:
        target_inputs = model_inputs
    else:
        target_inputs = to_list(target_inputs)
    model_outputs = to_list(model.output)

    if input_mask is None:
        # keep every input
        # input_mask=[input_.name for input_ in model_inputs]
        input_mask = []

    output_nodes = []
    for model_output in model_outputs:
        node_index = model_output._keras_history.node_index
        nodes = model._nodes_by_depth[node_index]
        try:
            nodes_ = [node for node in nodes if node.operation.output.name == model_output.name]
            node = nodes_[0]
            output_nodes.append(node)
        except IndexError:
            model_output_bis = model_output._keras_history.operation.output
            node_index_bis = model_output._keras_history.operation.output._keras_history.node_index
            nodes_bis = model._nodes_by_depth[node_index_bis]
            nodes_bis_ = [
                node for node in nodes_bis if node.operation.output.name == model_output_bis.name
            ]
            node = nodes_bis_[0]
            output_nodes.append(node)

    # if gradient is None: create inputs for backward mask
    if gradient is None:
        gradient = [
            Input(
                list(output_i.shape[1:]),
                name="{}_{}".format(output_i.name, "gradient"),
            )
            for output_i in model_outputs
        ]
        grad_input = [True] * len(model_outputs)

    if isinstance(gradient, list):
        assert len(gradient) == len(
            output_nodes
        ), "Mismatch between gradient and output nodes: The gradient must be specified for every output node, or not specified at all."
    else:
        gradient = [gradient]

    gradient = [get_gradient(grad, model_inputs) for grad in gradient]

    outputs = []
    is_linear = True

    # for input_ in model_inputs:
    for input_ in target_inputs:
        input_name = input_.name
        if not input_name in input_mask:
            outputs_i = []
            is_linear_i = True

            for i, (grad, output_node) in enumerate(zip(gradient, output_nodes)):
                output_node, is_linear_node, keep_branch = get_backward_node(
                    output_node,
                    grad,
                    mapping_keras2backward_classes,
                    input_name=input_name,
                    get_backward=get_backward,
                )
                if keep_branch:
                    outputs_i.append(output_node)
                    is_linear_i = min(is_linear_i, is_linear_node)

            if not len(outputs_i):
                continue
            else:
                outputs += outputs_i
                is_linear = min(is_linear, is_linear_i)

    inputs = []
    # check if the model is linear
    if not is_linear:
        inputs = [
            inp for inp in model_inputs
        ]  # due to masking some inputs may be not required to produce the input (aka linear branches of the model). Up to now we have no way to track it
    inputs += extra_inputs

    inputs += [gradient[i] for i in range(len(model_outputs)) if grad_input[i]]

    if len(inputs) == 1:
        inputs = inputs[0]
    if len(outputs) == 1:
        outputs = outputs[0]

    # or dictionary
    return BackwardModel(inputs, outputs, n_input=len(model.inputs))
