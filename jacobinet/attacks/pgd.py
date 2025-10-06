from typing import List, Union

import keras
import keras.ops as K  # type:ignore
import numpy as np
from jacobinet.attacks.base_attacks import AdvLayer, AdvModel
from jacobinet.attacks.fgsm import get_fgsm_model
from jacobinet.utils import to_list
from keras import KerasTensor as Tensor  # type:ignore
from keras.layers import RNN, Layer  # type:ignore
from keras.losses import Loss  # type:ignore

from .utils import PGD, clip_lp_ball


# define a RNN Cell, as a layer subclass.
class PGD_Cell(keras.Layer):
    """
    A custom RNN cell that applies the Projected Gradient Descent (PGD) attack as part of the recurrent computation.

    This class defines a custom RNN cell that performs adversarial perturbations using the Projected Gradient Descent (PGD) method.
    The cell operates within the recurrent loop of an RNN, where at each timestep, it applies the PGD attack to the input
    using a pre-defined `fgsm_model` and returns the adversarially perturbed input. The perturbation is then clipped to a defined range.

    Parameters:
        fgsm_model (AdvModel): The Fast Gradient Sign Method (FGSM) model used to compute adversarial perturbations.
        **kwargs: Additional keyword arguments to pass to the `keras.Layer` constructor.

    Attributes:
        state_size (int): The number of states maintained by the cell. In this case, it is fixed at 1.


    Example:
        # Example usage within an RNN layer
        pgd_cell = PGD_Cell(fgsm_model=fgsm_model)
        rnn_layer = RNN(pgd_cell)
        output = rnn_layer(inputs)
    """

    def __init__(self, fgsm_model, epsilon: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.fgsm_model = fgsm_model
        self.state_size = 1
        self.epsilon = epsilon

    def build(self, input_shape):
        self.built = True

    def call(self, y, states):
        x = states[0]  # x
        x_init = states[1]
        if len(states) > 2:
            lower = states[2]
            upper = states[3]
        # get adversarial attack using fgsm_model
        adv_x = self.fgsm_model([x, y] + list(states[2:]))
        adv_x = clip_lp_ball(adv_x, x_init, self.epsilon, p=self.fgsm_model.p)

        if len(states) > 2:
            adv_x = K.maximum(adv_x, lower)
            adv_x = K.minimum(adv_x, upper)
        return adv_x, [adv_x, x_init] + list(states[2:])


class ProjectedGradientDescent(AdvLayer):
    """
    A custom Keras layer that implements the Projected Gradient Descent (PGD) adversarial attack.

    This class applies the PGD method to generate adversarial examples by iterating over a sequence of
    adversarial perturbations. It uses an internal `FGSM` model and the `PGD_Cell` to apply the attack at
    each iteration. The cell is embedded in an `RNN` to iteratively refine the adversarial examples.

    Parameters:
        n_iter (int): The number of iterations for the PGD attack. Default is 10.
        fgsm_model (AdvModel): The Fast Gradient Sign Method (FGSM) model used to compute adversarial perturbations.
        **kwargs: Additional keyword arguments passed to the `AdvLayer` constructor.

    Example:
        # Example usage within a model
        pgd_layer = ProjectedGradientDescent(fgsm_model=fgsm_model, n_iter=10)
        model_with_pgd = keras.models.Model(inputs=model.inputs, outputs=pgd_layer(model.outputs))
    """

    def __init__(
        self,
        n_iter: int = 10,
        epsilon: float = 0,
        fgsm_model: AdvModel = None,
        random_init: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_iter = n_iter
        self.epsilon = epsilon
        self.fgsm_layer = fgsm_model.layer_adv
        pgd_cell = PGD_Cell(fgsm_model=fgsm_model, epsilon=self.epsilon)
        self.inner_layer = RNN(pgd_cell, return_sequences=True)

        # init attributes with fgsm_layer

        # self.upper = self.fgsm_layer.upper
        # self.lower = self.fgsm_layer.lower
        self.p = self.fgsm_layer.p
        # self.radius = self.fgsm_layer.radius

        self.random_init = random_init

    # saving
    def get_config(self):
        config = super().get_config()
        inner_layer_config = keras.saving.serialize_keras_object(self.inner_layer)
        config["pgd_cell"] = inner_layer_config
        config["n_iter"] = self.n_iter

        return config

    def build(self, input_shape):
        # build fgsm model
        self.built = True

    def set_upper(self, upper):
        self.fgsm_layer.set_upper(upper)
        self.upper = self.fgsm_layer.upper

    def set_lower(self, lower):
        self.fgsm_layer.set_lower(lower)
        self.lower = self.fgsm_layer.lower

    def set_p(self, p):
        self.fgsm_layer.set_p(p)
        self.p = self.fgsm_layer.p

    def compute_output_shape(self, input_shape):
        input_dim_with_batch: tuple[int] = input_shape[0]
        return (input_dim_with_batch[0], self.n_iter) + input_dim_with_batch[1:]

    def call(self, inputs, training=None, mask=None):
        x, y = inputs[:2]
        z = keras.ops.repeat(keras.ops.expand_dims(y, 1), self.n_iter, 1)

        if self.random_init:
            # start with a noisy version around x
            input_shape_wo_batch = x.shape[1:]
            noise = keras.random.uniform(
                input_shape_wo_batch,
                minval=-self.epsilon,
                maxval=self.epsilon,
                dtype=None,
                seed=None,
            )
            x_0 = x + K.expand_dims(noise, 0)
            # clip
            # x_0 = K.maximum(x_0, x - self.radius)
            # x_0 = K.minimum(x_0, x + self.radius)
            # clip ball
            x_0 = K.maximum(x_0, self.lower)
            x_0 = K.minimum(x_0, self.upper)
        else:
            x_0 = x
        output = self.inner_layer(z, initial_state=[x_0, x] + inputs[2:])

        return output


def get_pgd_model(
    model,
    loss: Union[str, Loss, Layer] = "categorical_crossentropy",
    epsilon: float = 0,
    p: float = np.inf,
    alpha=2 / 255,
    n_iter=10,
    random_init: bool = False,
    mapping_keras2backward_classes={},  # define type
    mapping_keras2backward_losses={},
    **kwargs,
) -> AdvModel:  # we do not compute gradient on extra_inputs, loss should return (None, 1)
    """
    Creates an adversarial model that applies the Projected Gradient Descent (PGD) attack using an existing model.

    This function constructs a model that applies the PGD attack to the input data by iterating over adversarial perturbations.
    It first creates an FGSM model, then uses the PGD attack to iteratively generate adversarial examples,
    and finally computes the adversarial loss using categorical cross-entropy.

    Parameters:
        model: The base Keras model on which adversarial attacks will be applied.
        loss: The loss function to be used for training the model.
            Defaults to 'categorical_crossentropy'. Can be a string, a Keras Loss object, or a Keras Layer.
        mapping_keras2backward_classes (dict): A dictionary to map Keras classes to their corresponding backward classes.
            Default is an empty dictionary.
        mapping_keras2backward_losses: A dictionary to map Keras losses to their corresponding backward losses.
            Default is an empty dictionary.
        **kwargs: Additional arguments passed to the `get_fgsm_model` and `ProjectedGradientDescent` functions.

    Returns:
        AdvModel: A model that applies the PGD attack to generate adversarial examples and compute the loss.

    Example:
        # Example usage:
        pgd_model = get_pgd_model(model, loss='categorical_crossentropy', n_iter=20)
        pgd_model.compile(optimizer='adam', loss='categorical_crossentropy')
    """

    extra_inputs = []
    if "extra_inputs" in kwargs:
        extra_inputs = kwargs["extra_inputs"]

        kwargs.pop("extra_inputs")  # remove extra_inputs so it is not parsed by fgsm

    bounds = []
    if "upper" in kwargs:
        extra_inputs = [kwargs["lower"], kwargs["upper"]]
        bounds = extra_inputs

    fgsm_model = get_fgsm_model(
        model,
        loss=loss,
        p=p,
        epsilon=alpha,
        mapping_keras2backward_classes=mapping_keras2backward_classes,  # define type
        mapping_keras2backward_losses=mapping_keras2backward_losses,
    )
    inputs: List[Tensor] = to_list(fgsm_model.inputs)
    if len(extra_inputs):
        # remove upper and lower from inputs
        inputs = inputs[:-2]

    pgd_layer = ProjectedGradientDescent(
        epsilon=epsilon,
        n_iter=n_iter,
        fgsm_model=fgsm_model,
        random_init=random_init,
        p=p,
        **kwargs,
    )

    output_adv = pgd_layer(inputs + bounds)

    if "return_best" in kwargs:
        # filter with the most adversarial example
        input_shape_wo_batch = list(inputs[0].shape[1:])
        pred_adv = K.reshape(output_adv, [-1] + input_shape_wo_batch)
        y_adv = model(pred_adv)

        n_class = y_adv.shape[-1]
        y_gt = K.repeat(K.expand_dims(inputs[1], 1), n_iter, 1)  # (batch, n_iter, n_class)
        y_gt = K.reshape(y_gt, [-1, n_class])
        # compute cross entropy
        loss_adv = K.reshape(
            K.categorical_crossentropy(y_gt, y_adv, from_logits=True), [-1, n_iter]
        )
        index_adv = K.argmax(loss_adv, -1)[:, None]  # (batch, 1)
        mask = K.one_hot(index_adv, n_iter)  # (batch, n_iter)
        mask = K.reshape(mask, [-1, n_iter] + [1] * len(input_shape_wo_batch))

        output = K.sum(mask * output_adv, 1)
    else:
        output = output_adv[:, -1]

    pgd_model = AdvModel(
        inputs=inputs + extra_inputs,
        outputs=output,
        layer_adv=pgd_layer,
        backward_model=fgsm_model.backward_model,
        method=PGD,
    )

    return pgd_model
