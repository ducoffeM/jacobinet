import keras
import keras.ops as K
import numpy as np
import pytest
from jacobinet import get_backward_layer as get_backward
from keras.layers import Input
from keras.models import Model, Sequential
from keras_custom.layers.numpy import *

from .conftest import compute_backward_layer


def _test_backward(layer, clip_min=-np.inf, clip_max=np.inf):
    input_shape = (32,)
    model = Sequential([layer])
    model(K.ones([1] + list(input_shape), dtype=keras.config.floatx()))

    backward_layer = get_backward(layer)

    mask_output = Input(input_shape)
    input_ = Input(input_shape)
    output = backward_layer([mask_output, input_])
    model_backward = Model([mask_output, input_], output)

    compute_backward_layer(
        input_shape, model, model_backward, input_random=True, clip_min=clip_min, clip_max=clip_max
    )


@pytest.mark.parametrize(
    "keras_layer",
    [
        Abs(),
        Absolute(),
        # AMax(axis=-1, keepdims=True),
        # AMin(axis=-1, keepdims=True),
        Arccos(),
        Arcsinh(),
        Arctan(),
        # Arctanh(),
        # Average(axis=-1),
        Cos(),
        Cosh(),
        # Cumprod(axis=-1),
        # Cumsum(axis=-1),
        # Diagonal(axis1=1, axis2=2),
        # ExpandDims(axis=-1),
        # Expm1(),
        # Floor(),
        # FullLike(3),
        # MoveAxis(2, 1),
        # Negative(),
        ## Norm(ord=2, axis=-1),
        # OnesLike(),
        # Prod(axis=-1, keepdims=True),
        # Reciprocal(),
        # Repeat(repeats=3, axis=-1),
        # Roll(shift=2, axis=-1),
        # Sign(),
        # Sin(),
        # Sinh(),
        # Sort(axis=-1),
        Square(),
        # SwapAxes(axis1=1, axis2=2),
        # Tan(),
        # Trace(),
        # Transpose(axes=(0, 2, 1)),
        # Tril(k=2),
        # Triu(k=1),
        # Trunc(),
        # Var(axis=-1),
        # ZerosLike(),
    ],
)
def test_backward_unary_ops(keras_layer):
    _test_backward(keras_layer)


@pytest.mark.parametrize(
    "keras_layer, clip_min, clip_max",
    [
        (Arccosh(), 1.1, np.inf),
        (Arcsin(), -0.9, 0.9),
        (Arctanh(), -0.9, 0.9),
        # Average(axis=-1),
        # Cos(),
        # Cosh(),
        # Cumprod(axis=-1),
        # Cumsum(axis=-1),
        # Diagonal(axis1=1, axis2=2),
        # ExpandDims(axis=-1),
        # Expm1(),
        # Floor(),
        # FullLike(3),
        (Log(), 1.0, 100.0),
        (Log10(), 1, 100.0),
        (Log1p(), 0, 100.0),
        (Log2(), 1.0, 100.0),
        # MoveAxis(2, 1),
        # Negative(),
        ## Norm(ord=2, axis=-1),
        # OnesLike(),
        # Prod(axis=-1, keepdims=True),
        # Reciprocal(),
        # Repeat(repeats=3, axis=-1),
        # Roll(shift=2, axis=-1),
        # Sign(),
        # Sin(),
        # Sinh(),
        # Sort(axis=-1),
        (Sqrt(), -3, -1),
        (Sqrt(), 0.1, 100),
        # Square(),
        # SwapAxes(axis1=1, axis2=2),
        (Tan(), -1.0, 1.0),
        # Trace(),
        # Transpose(axes=(0, 2, 1)),
        # Tril(k=2),
        # Triu(k=1),
        # Trunc(),
        # Var(axis=-1),
        # ZerosLike(),
    ],
)
def test_backward_unary_ops_arccosh(keras_layer, clip_min, clip_max):
    _test_backward(keras_layer, clip_min=clip_min, clip_max=clip_max)
