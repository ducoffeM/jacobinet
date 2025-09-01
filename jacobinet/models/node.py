from typing import Callable, List, Optional, Union

import numpy as np
from jacobinet.layers.convert import get_backward as get_backward_layer
from jacobinet.layers.layer import BackwardLayer, BackwardLinearLayer
from jacobinet.models.utils import FuseGradients, to_list
from keras import KerasTensor as Tensor  # type:ignore
from keras.layers import InputLayer, Layer  # type:ignore
from keras.models import Model, Sequential  # type:ignore
from keras.src.ops.node import Node


def get_backward_node(
    node: Node,
    gradient: Tensor,
    mapping_keras2backward_classes: Optional[dict[type[Layer], type[BackwardLayer]]] = None,
    input_name=None,
    get_backward: Callable = get_backward_layer,
):
    """
    Recursively computes the backward gradients for a given node in a computational graph.

    This function is responsible for traversing the computational graph starting from a specific node and calculating
    the gradients for that node. It considers both the structure of the model (layer-wise) and the operations performed
    by each layer. Depending on the layer type, it computes the gradients and returns them in the correct format.
    The function supports layers like `InputLayer`, `Sequential`, `Model`, and custom layers defined as `BackwardLayer`.

    Args:
        node: The current node in the computational graph whose gradients need to be computed.
        gradient: The gradient to be propagated backward through the network, starting from this node.
        mapping_keras2backward_classes: A dictionary that maps Keras layer types to their corresponding
            backward layer classes. This is used to obtain the backward-compatible version of the layer for gradient propagation.
        input_name: The name of the input layer to track. If provided, the function will ensure that gradients
            are computed for the corresponding input layer.
        get_backward: A function to get the backward version of the layer. Defaults to `get_backward_layer`.

    Returns:
        tuple: A tuple containing:
            - `output`: The computed backward gradients for the current node.
            - `is_linear`: Indicates whether the backward computation is linear (can be represented by a simple chain of operations).
            - `keep_output`: Indicates whether the computed gradients should be propagated further.

    Raises:
        ValueError: If unsupported layer types are encountered.
        NotImplementedError: If certain layer types or operations are not implemented for backward computation.

    Example:
        ```python
        output, is_linear, keep_output = get_backward_node(node, gradient, mapping_keras2backward_classes)
        ```
    """
    # step 1: get parents
    parent_nodes: List[None] = node.parent_nodes

    # step 2: get layer from node
    layer_node: Union[Layer, Model, Sequential] = node.operation

    # step 3.1 if layer is an InputLayer stop the algorithm
    if isinstance(layer_node, InputLayer):
        # check we reach the right input
        if input_name is None:
            return gradient, True, True
        else:
            return gradient, True, layer_node.output.name == input_name

    # step 4: get backward layer
    backward_layer_node: BackwardLayer = get_backward(
        layer_node,
        mapping_keras2backward_classes=mapping_keras2backward_classes,
    )

    is_linear = True
    if isinstance(backward_layer_node, BackwardLinearLayer):
        gradients = backward_layer_node(gradient)
    elif isinstance(layer_node, Sequential):
        if backward_layer_node.is_linear:
            gradients = backward_layer_node(gradient)
        else:
            is_linear = False
            layer_node_inputs = to_list(layer_node.input)
            gradients = backward_layer_node(layer_node_inputs + [gradient])
    elif isinstance(layer_node, Model):
        if backward_layer_node.is_linear:
            gradients = backward_layer_node(gradient)
        else:
            is_linear = False
            # warning: in case of nested models the following can happen: node.operation.input != node.parent_nodes[0].operation.output
            # consider working on the parents'outputs to avoid disconnected graph
            layer_node_inputs = []
            for parent_node in parent_nodes:
                layer_node_inputs = to_list(parent_node.operation.output)
            gradients = backward_layer_node(layer_node_inputs + [gradient])
    else:
        layer_node_inputs = to_list(layer_node.input)
        gradients = backward_layer_node([gradient] + layer_node_inputs)
        is_linear = False

    keep_output = True
    # get input_name of the layer
    layer_inputs = to_list(layer_node.input)
    if input_name in [e.name for e in layer_inputs]:
        if isinstance(gradients, list):
            # do something
            index_grad = np.where([e.name == input_name for e in layer_inputs])[0]
            return gradients[index_grad], is_linear, keep_output
        else:
            return gradients, is_linear, keep_output

    if backward_layer_node.n_input != 1:
        results = [
            get_backward_node(
                p_node,
                p_grad,
                mapping_keras2backward_classes,
                input_name,
                get_backward=get_backward,
            )
            for (p_node, p_grad) in zip(parent_nodes, gradients)
        ]

        results = [
            r for r in results if r[-1]
        ]  # keep the parents whom is connected to the right input

        if len(results):
            outputs = [
                r[0] for r in results
            ]  # output tensor is the first output of get_backward_node

            if is_linear:
                is_linear = min(is_linear, min([r[1] for r in results]))
            # chain rule along multiple inputs: expand, concat and sum
            fuse_layer = FuseGradients()
            output = fuse_layer(outputs)
        else:
            output = None
            is_linear = False
            keep_output = False
    else:
        result = get_backward_node(
            parent_nodes[0],
            gradients,
            mapping_keras2backward_classes,
            input_name,
            get_backward=get_backward,
        )
        output = result[0]
        keep_output = result[-1]
        if is_linear:
            is_linear = min(is_linear, result[1])

    return output, is_linear, keep_output
