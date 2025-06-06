import keras
import numpy as np
import pytest
from jacobinet import get_backward_layer as get_backward
from keras.layers import (
    Cropping1D,
    Cropping2D,
    Cropping3D,
    Flatten,
    Permute,
    RepeatVector,
    Reshape,
    UpSampling1D,
    UpSampling2D,
    UpSampling3D,
    ZeroPadding1D,
    ZeroPadding2D,
)
from keras.models import Sequential

from .conftest import is_invertible, linear_mapping, serialize


def test_backward_Reshape():
    keras.config.set_image_data_format(data_format="channels_first")
    input_shape = (2, 5, 10)
    # data_format == 'channels_first'
    layer = Reshape((2, 50))
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    is_invertible(layer, backward_layer)
    linear_mapping(layer, backward_layer)
    serialize(layer, backward_layer)
    backward_layer = get_backward(layer, use_bias=True)
    linear_mapping(layer, backward_layer)


def test_backward_RepeatVector():
    keras.config.set_image_data_format("channels_first")
    input_shape = (10,)
    layer = RepeatVector(2)
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    is_invertible(layer, backward_layer)
    linear_mapping(layer, backward_layer)
    serialize(layer, backward_layer)
    backward_layer = get_backward(layer, use_bias=True)
    linear_mapping(layer, backward_layer)


def test_backward_Permute():
    keras.config.set_image_data_format("channels_first")
    input_shape = (2, 5, 10)
    # data_format == 'channels_first'
    layer = Permute((1, 3, 2))
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    is_invertible(layer, backward_layer)
    linear_mapping(layer, backward_layer)

    serialize(layer, backward_layer)
    backward_layer = get_backward(layer, use_bias=True)
    linear_mapping(layer, backward_layer)


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_backward_Flatten(data_format):
    keras.config.set_image_data_format(data_format)
    if data_format == "channels_first":
        input_shape = (2, 4, 3)
    else:
        input_shape = (4, 3, 2)
    # data_format == 'channels_first'
    layer = Flatten()
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    is_invertible(layer, backward_layer)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_backward_Cropping2D(data_format):
    keras.config.set_image_data_format(data_format)
    if data_format == "channels_first":
        input_shape = (2, 12, 11)
    else:
        input_shape = (12, 11, 2)
    # data_format == 'channels_first'
    layer = Cropping2D(cropping=(3, 3))
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_backward_ZeroPadding2D(data_format):
    keras.config.set_image_data_format(data_format)
    if data_format == "channels_first":
        input_shape = (2, 12, 11)
    else:
        input_shape = (12, 11, 2)
    # data_format == 'channels_first'
    layer = ZeroPadding2D(padding=(3, 3))
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_backward_Cropping1D(data_format):
    keras.config.set_image_data_format(data_format)
    input_shape = (11, 2)
    # data_format == 'channels_first'
    layer = Cropping1D(cropping=3)
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_backward_ZeroPadding1D(data_format):
    keras.config.set_image_data_format(data_format)
    if data_format == "channels_first":
        input_shape = (2, 12)
    else:
        input_shape = (12, 2)
    # data_format == 'channels_first'
    layer = ZeroPadding1D(padding=3, data_format="channels_first")
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_backward_Cropping3D(data_format):
    keras.config.set_image_data_format(data_format)
    if data_format == "channels_first":
        input_shape = (2, 12, 11, 10)
    else:
        input_shape = (12, 11, 10, 2)
    # data_format == 'channels_first'
    layer = Cropping3D(cropping=(3, 3, 2))
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_backward_UpSampling2D(data_format):
    keras.config.set_image_data_format(data_format)
    if data_format == "channels_first":
        input_shape = (2, 12, 11)
    else:
        input_shape = (12, 11, 2)
    # data_format == 'channels_first'
    layer = UpSampling2D(size=(2, 2))
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_backward_UpSampling1D(data_format):
    keras.config.set_image_data_format(data_format)
    if data_format == "channels_first":
        input_shape = (2, 12)
    else:
        input_shape = (12, 2)
    # data_format == 'channels_first'
    layer = UpSampling1D(size=3)
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)

    # data_format == 'channels_first'
    layer = UpSampling1D(size=2)
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_backward_UpSampling3D(data_format):
    keras.config.set_image_data_format(data_format)
    pytest.skip("skip tests for 3D")
    input_shape = (2, 12, 11, 10)

    # data_format == 'channels_first'
    layer = UpSampling3D(size=(3, 3, 2), data_format="channels_first")
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])

    backward_layer = get_backward(layer, use_bias=False)

    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)
