from typing import Any, List, Tuple, Union

import keras  # type:ignore
import keras.ops as K  # type:ignore
import numpy as np
from jacobinet import clone_to_backward
from jacobinet.losses import deserialize, get_loss_as_layer
from jacobinet.models import BackwardModel
from jacobinet.utils import to_list
from keras import KerasTensor as Tensor  # type:ignore
from keras.layers import Input, Layer  # type:ignore
from keras.losses import Loss  # type:ignore
from keras.models import Model, Sequential  # type:ignore

from .utils import FGSM


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


def get_adv_model_base(
    model,
    loss: Union[str, Layer] = "categorical_crossentropy",
    attack: str = FGSM,
    mapping_keras2backward_classes={},
    mapping_keras2backward_losses={},
    **kwargs,
) -> Model:  # we do not compute gradient on extra_inputs, loss should return (None, 1)
    """
    Creates a base adversarial model by incorporating the specified attack and loss function.

    This function returns a model that computes adversarial examples using the specified attack
    method and loss function. The attack is applied during the backward pass of the model.

    Args:
        model: A Keras model (either a `Model` or `Sequential` instance) to be used for generating adversarial examples.
        loss: The loss function used in the model. It can be a string (e.g., 'categorical_crossentropy'), a `Layer` object,
              or a Keras `Loss` object. Defaults to 'categorical_crossentropy'.
        attack: The attack method used to create adversarial examples. Defaults to FGSM (Fast Gradient Sign Method).
        mapping_keras2backward_classes: A dictionary mapping Keras layers to their backward counterparts for gradient computation.
        mapping_keras2backward_losses: A dictionary mapping loss functions to their backward counterparts.
        **kwargs: Additional arguments passed to the `get_model_with_loss` function, such as `gt_shape` or other layer-specific settings.

    Returns:
        Model: A Keras `Model` that computes the adversarial loss during training or evaluation. This model includes
               the attack method and loss function, but does not compute gradients for extra inputs.

    Raises:
        NotImplementedError: If the model has multiple inputs or outputs, as this function currently supports single input-output models.

    Example:
        adv_model = get_adv_model_base(model=my_model, loss='categorical_crossentropy', attack='FGSM')
        adv_model.compile(optimizer='adam', loss='categorical_crossentropy')
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

    backward_model_base_attack = clone_to_backward(
        model=model_with_loss,
        mapping_keras2backward_classes=mapping_keras2backward_classes,
        gradient=keras.Variable(np.ones((1, 1), dtype="float32")),
        input_mask=input_mask,
    )
    # convert it into an AdvModel
    return backward_model_base_attack


class AdvLayer(Layer):
    """
    A custom Keras layer for adversarial perturbations.

    This layer is designed to apply adversarial perturbations to the input data, typically used in adversarial training
    or for creating adversarial examples. The perturbations can be controlled by various parameters, such as the
    magnitude (`epsilon`), the type of norm (`p`), and constraints on the perturbation (`lower`, `upper`, and `radius`).

    Example:
        adv_layer = AdvLayer(epsilon=0.1, p=2, radius=1.0)
        perturbed_input = adv_layer(input_tensor)

    """

    def __init__(
        self,
        epsilon: float = 0.0,
        lower: float = -np.inf,
        upper: float = np.inf,
        p: float = -1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epsilon = keras.Variable(epsilon, trainable=False)
        self.lower = lower
        self.upper = upper
        self.p = p

    # @keras.ops.custom_gradient
    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError()

    # saving
    def get_config(self):
        config = super().get_config()
        eps_config = keras.saving.serialize_keras_object(self.epsilon)
        config["epsilon"] = eps_config
        config["lower"] = self.lower
        config["upper"] = self.upper
        config["radius"] = self.radius
        return config

    def set_upper(self, upper):
        self.upper = upper

    def set_lower(self, lower):
        self.lower = lower

    def set_p(self, p):
        self.p = p

    def set_radius(self, radius):
        self.radius = radius

    def get_upper(self):
        return self.upper

    def get_lower(self):
        return self.lower

    def get_p(self):
        return self.p

    def get_radius(self):
        return self.radius


class AdvModel(keras.Model):
    """
    A custom Keras model that incorporates adversarial perturbations during training or inference.

    This model wraps around a Keras model and an adversarial layer (such as an `AdvLayer`) to apply adversarial
    attacks during forward propagation. The attack can be specified using the `method` parameter (e.g., FGSM, etc.).
    It allows the application of adversarial perturbations during training to improve model robustness.

    Example:
        adv_layer = AdvLayer(epsilon=0.1, p=2, radius=1.0)
        backward_model = BackwardModel(model)
        adv_model = AdvModel(layer_adv=adv_layer, backward_model=backward_model, method='fgsm')
        adv_model.set_upper(1.0)
        adv_model.set_lower(-1.0)

    """

    def __init__(
        self,
        layer_adv: AdvLayer,
        backward_model: BackwardModel,
        method="fgsm",  # replace by Enum
        radius=np.inf,
        p=np.inf,
        *args,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.backward_model = backward_model
        self.method = method
        self.layer_adv = layer_adv
        self.radius = radius
        self.p = p

    def get_config(self):
        config = super().get_config()
        backward_config = keras.saving.serialize_keras_object(self.backward_model)
        layer_config = keras.saving.serialize_keras_object(self.layer_adv)
        config["backward_model"] = backward_config
        config["method"] = self.method
        config["layer_adv"] = layer_config
        return config

    def set_upper(self, upper):
        self.layer_adv.set_upper(upper)

    def set_lower(self, lower):
        self.layer_adv.set_lower(lower)

    def set_p(self, p):
        self.layer_adv.set_p(p)

    def get_upper(self):
        return self.layer_adv.upper

    def get_lower(self):
        return self.layer_adv.get_lower()

    def get_p(self, p):
        return self.layer_adv.get_p()

    def compute_output_shape(self, input_shape):
        return input_shape[0]
