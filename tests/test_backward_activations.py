import keras
import keras.ops as K
import pytest
from jacobinet import get_backward_layer as get_backward
from keras.layers import ELU, Activation, Input, LeakyReLU, PReLU, ReLU
from keras.models import Model, Sequential

from .conftest import compute_backward_layer, serialize_model


def _test_backward_activation(layer):
    input_shape = (32,)
    model = Sequential([layer])
    model(K.ones([1] + list(input_shape), dtype=keras.config.floatx()))

    backward_layer = get_backward(layer)

    mask_output = Input(input_shape)
    input_ = Input(input_shape)
    output = backward_layer([mask_output, input_])
    model_backward = Model([mask_output, input_], output)

    compute_backward_layer(input_shape, model, model_backward)


def test_backward_ReLU():
    layer = ReLU(threshold=1.0, negative_slope=0.3)
    _test_backward_activation(layer)

    layer = ReLU(threshold=1.1, negative_slope=0.3)
    _test_backward_activation(layer)

    layer = ReLU(threshold=0, negative_slope=0.3, max_value=0.5)
    _test_backward_activation(layer)


def test_backward_ELU():
    layer = ELU()
    _test_backward_activation(layer)


def test_backward_LeakyReLU():
    layer = LeakyReLU()
    _test_backward_activation(layer)


def test_backward_PReLU():
    layer = PReLU()
    _test_backward_activation(layer)


@pytest.mark.parametrize(
    "activation_name",
    [
        "relu",
        "relu6",
        "leaky_relu",
        "elu",
        "selu",
        "softplus",
        "softsign",
        "sigmoid",
        "tanh",
        "silu",
        "exponential",
        "hard_sigmoid",
        "hard_silu",
        "linear",
        "mish",
        "swish",
        "hard_swish",
    ],
)
def test_backward_Activation(activation_name):
    layer = Activation(activation_name)
    _test_backward_activation(layer)
