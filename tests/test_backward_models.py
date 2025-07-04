import keras
import numpy as np
import pytest
import torch

# from jacobinet.models.sequential import get_backward_sequential
# from jacobinet.models.model import get_backward_functional
from jacobinet.models import clone_to_backward
from keras.layers import (
    AveragePooling2D,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Input,
    MaxPooling2D,
    ReLU,
    Reshape,
)
from keras.models import Model, Sequential

from .conftest import compute_backward_model, compute_output, serialize_model

# preliminary tests: gradient is derived automatically by considering single output model


def test_sequential_linear():
    keras.config.set_image_data_format("channels_first")
    input_dim = 32
    layers = [Dense(2, use_bias=False), Dense(1, use_bias=False)]
    model = Sequential(layers)
    _ = model(torch.ones((1, input_dim)))
    backward_model = clone_to_backward(model)
    # model is linear

    _ = backward_model(np.ones((1, 1)))

    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([32], model)
    serialize_model([1], backward_model)


def test_sequential_nonlinear():
    keras.config.set_image_data_format("channels_first")
    input_dim = 32
    layers = [Dense(2, use_bias=False), ReLU(), Dense(1, use_bias=False)]
    model = Sequential(layers)
    _ = model(torch.ones((1, input_dim)))
    backward_model = clone_to_backward(model)
    # model is not linear
    _ = backward_model([np.ones((1, input_dim)), np.ones((1, 1))])
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([input_dim, 1], backward_model)


def test_sequential_multiD():
    keras.config.set_image_data_format("channels_first")
    input_dim = 36
    layers = [
        Reshape((1, 6, 6)),
        Conv2D(2, (3, 3)),
        ReLU(),
        Reshape((-1,)),
        Dense(1),
    ]
    model = Sequential(layers)
    _ = model(torch.ones((1, input_dim)))
    backward_model = clone_to_backward(model)
    # model is not linear
    _ = backward_model([torch.ones((1, input_dim)), torch.ones((1, 1))])
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([input_dim, 1], backward_model)


@pytest.mark.parametrize(
    "padding, layer_name",
    [
        ("valid", "average"),
        ("same", "average"),
        ("valid", "max"),
        ("same", "max"),
    ],
)
def test_sequential_multiD_pooling(padding, layer_name):
    keras.config.set_image_data_format("channels_first")
    input_shape = (1, 3, 6)
    input_dim = np.prod(input_shape)
    layer_ = None
    if layer_name == "average":
        layer_ = AveragePooling2D((2, 2), (1, 1), padding=padding)
    if layer_name == "max":
        layer_ = MaxPooling2D((2, 2), (1, 2), padding=padding)
    layers = [Reshape(input_shape), layer_, ReLU(), Reshape((-1,)), Dense(1)]
    model = Sequential(layers)
    _ = model(torch.ones((1, input_dim)))

    backward_model = clone_to_backward(model)
    # model is not linear
    _ = backward_model([torch.ones((1, input_dim)), torch.ones((1, 1))])
    compute_backward_model((input_dim,), model, backward_model, value="rand")
    compute_backward_model((input_dim,), model, backward_model, value="zeros")
    compute_backward_model((input_dim,), model, backward_model, value="ones")
    serialize_model([input_dim, 1], backward_model)


def _test_sequential_multiD_channel_last():
    data_format = keras.config.get_image_data_format()
    input_dim = 72
    if data_format == "channels_first":
        target_shape = (2, 6, 6)
    else:
        target_shape = (6, 6, 2)
    layers = [
        Reshape(target_shape),
        DepthwiseConv2D(2, (3, 3)),
        ReLU(),
        Reshape((-1,)),
        Dense(1),
    ]
    model = Sequential(layers)
    _ = model(torch.ones((1, input_dim)))
    backward_model = clone_to_backward(model)
    # model is not linear
    _ = backward_model([torch.ones((1, input_dim)), torch.ones((1, 1))])
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([input_dim, 1], backward_model)


# same using model instead of Sequential
def test_model_linear():
    keras.config.set_image_data_format("channels_first")
    input_dim = 32
    layers = [Dense(2, use_bias=False), Dense(1, use_bias=False)]
    input_ = Input((input_dim,))
    output = None

    for layer in layers:
        if output is None:
            output = layer(input_)
        else:
            output = layer(output)

    model = Model(input_, output)
    _ = model(torch.ones((1, input_dim)))
    backward_model = clone_to_backward(model)
    # model is linear
    _ = backward_model(np.ones((1, 1)))
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([1], backward_model)


def test_model_nonlinear():
    keras.config.set_image_data_format("channels_first")
    input_dim = 32
    layers = [Dense(2, use_bias=False), ReLU(), Dense(1, use_bias=False)]
    input_ = Input((input_dim,))
    output = None
    for layer in layers:
        if output is None:
            output = layer(input_)
        else:
            output = layer(output)
    model = Model(input_, output)
    _ = model(torch.ones((1, input_dim)))
    backward_model = clone_to_backward(model)
    # model is not linear
    _ = backward_model([np.ones((1, input_dim)), np.ones((1, 1))])
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([input_dim, 1], backward_model)


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_model_multiD(data_format):
    keras.config.set_image_data_format(data_format)
    input_dim = 36
    if data_format == "channels_first":
        target_shape = (1, 6, 6)
    else:
        target_shape = (6, 6, 1)
    layers = [
        Reshape(target_shape),
        Conv2D(2, (3, 3)),
        ReLU(),
        Reshape((-1,)),
        Dense(1),
    ]
    input_ = Input((input_dim,))
    output = None
    for layer in layers:
        if output is None:
            output = layer(input_)
        else:
            output = layer(output)
    model = Model(input_, output)
    _ = model(torch.ones((1, input_dim)))
    backward_model = clone_to_backward(model)
    # model is not linear
    _ = backward_model([torch.ones((1, input_dim)), torch.ones((1, 1))])
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([input_dim, 1], backward_model)


def _test_model_multiD_channel_last():
    data_format = keras.config.get_image_data_format()
    input_dim = 72
    if data_format == "channels_first":
        target_shape = (2, 6, 6)
    else:
        target_shape = (6, 6, 2)
    layers = [
        Reshape(target_shape),
        Conv2D(2, (3, 3)),
        ReLU(),
        Reshape((-1,)),
        Dense(1),
    ]
    input_ = Input((input_dim,))
    output = None
    for layer in layers:
        if output is None:
            output = layer(input_)
        else:
            output = layer(output)
    model = Model(input_, output)
    _ = model(torch.ones((1, input_dim)))
    backward_model = clone_to_backward(model)
    # model is not linear
    _ = backward_model([torch.ones((1, input_dim)), torch.ones((1, 1))])
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([input_dim, 1], backward_model)


###### encode gradient as a KerasVariable #####
@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_model_multiD_with_gradient_set(data_format):
    keras.config.set_image_data_format(data_format)
    input_dim = 36
    if data_format == "channels_first":
        target_shape = (1, 6, 6)
    else:
        target_shape = (6, 6, 1)
    layers = [
        Reshape(target_shape),
        Conv2D(2, (3, 3)),
        ReLU(),
        Reshape((-1,)),
        Dense(1),
    ]
    gradient = keras.Variable(np.ones((1, 1), dtype="float32"))
    input_ = Input((input_dim,))
    output = None
    for layer in layers:
        if output is None:
            output = layer(input_)
        else:
            output = layer(output)
    model = Model(input_, output)
    _ = model(torch.ones((1, input_dim)))
    backward_model = clone_to_backward(model, gradient=gradient)
    # model is not linear
    _ = backward_model(torch.ones((1, input_dim)))
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([input_dim], backward_model)


# extra inputs
@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_model_multiD_extra_input(data_format):
    keras.config.set_image_data_format(data_format)
    input_dim = 36
    if data_format == "channels_first":
        target_shape = (1, 6, 6)
    else:
        target_shape = (6, 6, 1)
    layers = [
        Reshape(target_shape),
        Conv2D(2, (3, 3)),
        ReLU(),
        Reshape((-1,)),
        Dense(1),
    ]
    input_ = Input((input_dim,))
    output = None
    for layer in layers:
        if output is None:
            output = layer(input_)
        else:
            output = layer(output)
    model = Model(input_, output)
    _ = model(torch.ones((1, input_dim)))
    # gradient is the result of extra_inputs
    extra_input = Input((10,))
    gradient = keras.ops.max(extra_input, axis=-1)
    backward_model = clone_to_backward(model, gradient=gradient, extra_inputs=[extra_input])
    # model is not linear
    _ = backward_model([torch.ones((1, input_dim)), torch.ones((1, 10))])

    mask_output = torch.eye(10)
    for i in range(10):
        compute_backward_model(
            (input_dim,),
            model,
            backward_model,
            0,
            grad_value=mask_output[i][None],
        )

    serialize_model([input_dim, 10], backward_model)


# multiple outputs
### multi output neural network #####
@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_model_multiD_multi_output(data_format):
    keras.config.set_image_data_format(data_format)
    input_dim = 36
    if data_format == "channels_first":
        target_shape = (1, 6, 6)
    else:
        target_shape = (6, 6, 1)
    input_dim = 36
    layers = [
        Reshape(target_shape),
        Conv2D(2, (3, 3)),
        ReLU(),
        Reshape((-1,)),
        Dense(10),
    ]
    input_ = Input((input_dim,))
    output = None
    for layer in layers:
        if output is None:
            output = layer(input_)
        else:
            output = layer(output)
    model = Model(input_, output)
    _ = model(torch.ones((1, input_dim)))
    backward_model = clone_to_backward(model, gradient=Input((10,)))
    # model is not linear
    _ = backward_model([torch.ones((1, input_dim)), torch.ones((1, 10))])

    for i in range(10):
        compute_backward_model((input_dim,), model, backward_model, i)

    serialize_model([input_dim, 10], backward_model)  #


### multi output neural network #####
def _test_model_multiD_multi_outputs():
    data_format = keras.config.image_data_format()
    input_dim = 36
    if data_format == "channels_first":
        target_shape = (1, 6, 6)
    else:
        target_shape = (6, 6, 1)
    layers_0 = [
        Reshape(target_shape),
        Conv2D(2, (3, 3)),
        ReLU(),
        Reshape((-1,)),
        Dense(10),
    ]
    layers_1 = [
        Reshape(target_shape),
        Conv2D(2, (3, 3)),
        ReLU(),
        Reshape((-1,)),
        Dense(20),
    ]
    input_ = Input((input_dim,))
    output = None
    output_0 = compute_output(input_, layers_0)
    output_1 = compute_output(input_, layers_1)

    model = Model(input_, [output_0, output_1])
    _ = model(torch.ones((1, input_dim)))
    backward_model = clone_to_backward(model, gradient=[Input((10,)), Input((20,))])
    # model is not linear

    _ = backward_model([torch.ones((1, input_dim)), torch.ones((1, 10)), torch.ones((1, 20))])

    # freeze one model and computer backward on the other branch

    model_0 = Model(input_, output_0)
    backward_model_0 = clone_to_backward(
        model, gradient=[Input((10,)), keras.Variable(np.zeros((1, 20)))]
    )
    backward_model_0_bis = clone_to_backward(model_0)
    grad_0 = backward_model_0([torch.ones((1, input_dim)), torch.ones((1, 10))])
    grad_0_bis = backward_model_0_bis([torch.ones((1, input_dim)), torch.ones((1, 10))])
    # import pdb; pdb.set_trace()

    for i in range(10):
        compute_backward_model((input_dim,), model_0, backward_model_0_bis, i)

    mask_output = torch.eye(10)
    for i in range(10):
        compute_backward_model((input_dim,), model, backward_model, i)

    # serialize_model([input_dim, 10], backward_model)


# nested models
@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_nested_sequential_linear(data_format):
    keras.config.set_image_data_format(data_format)
    input_dim = 32
    layers = [Dense(2, use_bias=False), Dense(input_dim, use_bias=False)]
    inner_model = Sequential(layers)
    _ = inner_model(torch.ones((1, input_dim)))

    layers = [Dense(2, use_bias=False), Dense(input_dim, use_bias=False)]
    inner_model_bis = Sequential(layers)
    _ = inner_model_bis(torch.ones((1, input_dim)))

    model = Sequential([Dense(input_dim), inner_model, inner_model_bis, Dense(1)])
    _ = model(torch.ones((1, input_dim)))
    backward_model = clone_to_backward(model)
    # model is linear
    _ = backward_model(np.ones((1, 1)))

    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([32], model)
    serialize_model([1], backward_model)


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_nested_sequential_nonlinear_linear(data_format):
    keras.config.set_image_data_format(data_format)
    input_dim = 32
    layers = [Dense(input_dim), ReLU(), Dense(3)]
    nested_model = Sequential(layers)
    _ = nested_model(torch.ones((1, input_dim)))

    model = Sequential([nested_model, Dense(1)])
    _ = model(torch.ones((1, input_dim)))

    backward_model = clone_to_backward(model)

    # check nested
    # model is not linear
    _ = backward_model([np.ones((1, input_dim)), np.ones((1, 1))])

    backward_nested_model = backward_model.layers[-1]

    compute_backward_model((input_dim,), nested_model, backward_nested_model, 0)
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([input_dim, 1], backward_model)


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_nested_sequential_linear_nonlinear(data_format):
    keras.config.set_image_data_format(data_format)
    input_dim = 32
    layers = [Dense(input_dim), ReLU(), Dense(3)]
    nested_model = Sequential(layers)
    _ = nested_model(torch.ones((1, input_dim)))

    model = Sequential([nested_model, ReLU(), Dense(1)])
    _ = model(torch.ones((1, input_dim)))

    backward_model = clone_to_backward(model)

    _ = backward_model([np.ones((1, input_dim)), np.ones((1, 1))])
    compute_backward_model((input_dim,), model, backward_model)

    # not supported with torch backend: to check with other backends
    # serialize_model([input_dim, 1], backward_model)


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_nested_sequential_model(data_format):
    keras.config.set_image_data_format(data_format)
    input_dim = 32
    layers = [Dense(input_dim), ReLU(), Dense(3)]
    input_nested = Input((input_dim,))
    output_nested = input_nested
    for layer in layers:
        output_nested = layer(output_nested)

    nested_model = Model(input_nested, output_nested)
    _ = nested_model(torch.ones((1, input_dim)))

    model = Sequential([nested_model, ReLU(), Dense(1)])
    _ = model(torch.ones((1, input_dim)))

    backward_model = clone_to_backward(model)

    _ = backward_model([np.ones((1, input_dim)), np.ones((1, 1))])
    compute_backward_model((input_dim,), model, backward_model)


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_model_linear_nested_sequential(data_format):
    keras.config.set_image_data_format(data_format)
    input_dim = 32
    layer_nested = [Dense(input_dim), Dense(input_dim)]
    model_nested = Sequential(layer_nested)
    _ = model_nested(torch.ones((1, input_dim)))

    layers = [model_nested, Dense(2, use_bias=False), Dense(1, use_bias=False)]
    input_ = Input((input_dim,))
    output = None
    for layer in layers:
        if output is None:
            output = layer(input_)
        else:
            output = layer(output)
    model = Model(input_, output)
    _ = model(torch.ones((1, input_dim)))
    backward_model = clone_to_backward(model)
    # model is linear
    _ = backward_model(np.ones((1, 1)))
    compute_backward_model((input_dim,), model, backward_model)


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_nested_model_linear_nonlinear(data_format):
    keras.config.set_image_data_format(data_format)
    input_dim = 32
    layers = [Dense(input_dim), ReLU(), Dense(3)]
    z = Input((input_dim,))
    y = z
    for layer in layers:
        y = layer(y)
    nested_model = Model(z, y)
    # nested_model = Sequential(layers)
    _ = nested_model(torch.ones((1, input_dim)))

    x = Input((input_dim,))
    output = x
    for layer in [nested_model, ReLU(), Dense(1)]:
        output = layer(output)
    model = Model(x, output)
    _ = model(torch.ones((1, input_dim)))

    backward_model = clone_to_backward(model)
    _ = backward_model([np.ones((1, input_dim)), np.ones((1, 1))])
    compute_backward_model((input_dim,), model, backward_model)
