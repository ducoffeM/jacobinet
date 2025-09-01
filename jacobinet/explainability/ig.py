from typing import Union

import keras
from jacobinet.losses import get_backward_model_with_loss
from keras.layers import RNN, Input, Layer
from keras.models import Model


class IG_Cell(keras.Layer):
    """
    ...
    """

    def __init__(self, backward_model, **kwargs):
        super().__init__(**kwargs)
        self.backward_model = backward_model
        self.state_size = 1

    def build(self, input_shape):
        self.built = True

    def call(self, z, states):
        # z.shape (batch, 1...len(x.shape[1:]))
        x = states[0]  # x
        y = states[1]
        if len(states) > 2:
            baseline = states[2]
            input_ = x + z * (baseline - x)
        else:
            input_ = (1 - z) * x

        xai_map = self.backward_model([input_, y])

        return xai_map, states


class IG(Layer):
    def __init__(
        self,
        steps: int = 10,
        backward_model: Model = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.steps = steps
        self.backward_model: Model = (backward_model,)
        ig_cell = IG_Cell(backward_model=backward_model)
        self.inner_layer = RNN(ig_cell, return_sequences=True)

    # saving
    def get_config(self):
        config = super().get_config()
        inner_layer_config = keras.saving.serialize_keras_object(self.inner_layer)
        config["backward_cell"] = inner_layer_config
        config["steps"] = self.steps
        return config

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, training=None, mask=None):
        z = keras.ops.linspace(0, 1, self.steps)[None]  # (1, self.steps)
        x = inputs[0]
        n_dim_without_batch = len(x.shape[1:])
        z = keras.ops.reshape(z, [1, self.steps] + [1] * n_dim_without_batch)
        z = keras.ops.expand_dims(0.0 * x, 1)
        output = self.inner_layer(z, initial_state=inputs[:3])
        # apply integral, aka empirical mean along
        import pdb

        pdb.set_trace()


def get_integrated_gradient_model(
    model: Model,
    loss: Union[str, Layer] = "logits",
    steps: int = 10,
    use_custom_baseline: bool = False,
    mapping_keras2backward_classes={},
    mapping_keras2backward_losses={},
    **kwargs,
):
    backbone_model = get_backward_model_with_loss(
        model=model,
        loss=loss,
        mapping_keras2backward_classes=mapping_keras2backward_classes,
        mapping_keras2backward_losses=mapping_keras2backward_losses,
        **kwargs,
    )

    # create new input
    ig_model = IG(steps=steps, backward_model=backbone_model)

    return ig_model


"""

The Integrated Gradients method attributes the prediction of a model to its input features by integrating the gradients of the model's output with respect to the input along a path from a baseline input to the actual input.

The formula for Integrated Gradients is:

\[
\text{IntegratedGradients}_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial F(x' + \alpha \cdot (x - x'))}{\partial x_i} d\alpha
\]

Where:
- \( x \) is the input,
- \( x' \) is the baseline input (e.g., all zeros),
- \( F \) is the model's prediction function,
- \( i \) indexes the input features,
- \( \alpha \) is a scaling coefficient from 0 to 1.

In practice, this integral is approximated using a Riemann sum over m steps:

\[
\text{IntegratedGradients}_i(x) \approx (x_i - x'_i) \cdot \frac{1}{m} \sum_{k=1}^{m} \frac{\partial F\left(x' + \frac{k}{m} (x - x')\right)}{\partial x_i}
\]
"""
