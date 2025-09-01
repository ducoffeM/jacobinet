from typing import List, Optional, Union

from jacobinet.layers.convert import get_backward as get_backward_layer
from jacobinet.layers.layer import BackwardLayer
from jacobinet.losses import Loss_Layer, get_backward_loss
from jacobinet.models import BackwardModel, BackwardSequential
from jacobinet.models.model import get_backward_functional
from jacobinet.models.sequential import get_backward_sequential
from jacobinet.models.utils import to_list
from keras import KerasTensor as Tensor
from keras.layers import Input, Layer  # type:ignore
from keras.models import Model, Sequential  # type:ignore


def clone_to_backward(
    model: Model,
    gradient: Union[None, Tensor, List[Tensor]] = None,
    mapping_keras2backward_classes: Optional[dict[type[Layer], type[BackwardLayer]]] = None,
    extra_inputs: Union[List[Input]] = [],
    input_mask=None,
    target_inputs=None,
) -> Union[BackwardModel, BackwardSequential]:
    """
    Clones a Keras model into a backward model for gradient-based backward computations.

    This function takes a Keras model and creates a backward-compatible version of it that supports
    gradient-based computations. Depending on the structure of the model (Sequential or Functional),
    it returns an appropriate `BackwardModel` or `BackwardSequential` with backward layers.

    Args:
        model: The original Keras model to be cloned into a backward model.
        gradient: The gradient tensor(s) used for backward computations.
        mapping_keras2backward_classes: A mapping of Keras layer types to their corresponding
            backward layer classes (e.g., `dict[type[Layer], type[BackwardLayer]]`).
        extra_inputs: Additional inputs to the backward model.
        input_mask: A mask for the input layers to apply in backward computation.
        target_inputs: The target input layers for the backward computation.

    Returns:
        Union[BackwardModel, BackwardSequential]: A backward-compatible version of the original Keras model,
            either a `BackwardModel` or `BackwardSequential`, depending on the structure of the original model.

    Raises:
        ValueError: If the model contains unsupported layer types or structure.
        NotImplementedError: If certain features (e.g., input masking or target input handling) are not implemented.

    Example:
        ```python
        backward_model = clone_to_backward(original_model, gradient=some_gradient)
        ```
    """

    def get_backward(layer, mapping_keras2backward_classes, **kwargs):
        if isinstance(layer, Sequential):
            if not input_mask is None:
                input_mask_ = to_list(input_mask)
                if layer.input.name in input_mask_:
                    raise NotImplementedError()
            if not (target_inputs is None):
                raise NotImplementedError()  # to do
            return get_backward_sequential(
                model=layer,
                mapping_keras2backward_classes=mapping_keras2backward_classes,
                get_backward=get_backward,
            )
        elif isinstance(layer, Model):
            return get_backward_functional(
                model=layer,
                mapping_keras2backward_classes=mapping_keras2backward_classes,
                input_mask=input_mask,
                get_backward=get_backward,
                target_inputs=target_inputs,
            )
        elif isinstance(layer, Loss_Layer):
            # keras losses automatically cast as Loss_Layer
            return get_backward_loss(layer)
        elif isinstance(layer, Layer):
            return get_backward_layer(layer, mapping_keras2backward_classes)
        else:
            raise ValueError("unsupported object layer={}".format(layer))

    if isinstance(model, Sequential):
        if not input_mask is None:
            raise NotImplementedError()
        backward_model = get_backward_sequential(
            model,
            gradient,
            mapping_keras2backward_classes,
            extra_inputs,
            get_backward=get_backward,
        )
        # backward_model.set_model(model)
    else:
        backward_model = get_backward_functional(
            model,
            gradient,
            mapping_keras2backward_classes,
            extra_inputs,
            input_mask,
            get_backward=get_backward,
            target_inputs=target_inputs,
        )
    return backward_model
