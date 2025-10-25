import keras
import keras.ops as K
import numpy as np
import pytest
from jacobinet import get_backward_layer as get_backward
from keras.layers import Input
from keras.models import Model, Sequential
from keras_custom.layers.numpy import *

from .conftest import compute_backward_layer


def _test_backward(layer, input_shape=(3,), clip_min=-np.inf, clip_max=np.inf):
    model = Sequential([layer])
    output = model(K.ones([1] + list(input_shape), dtype=keras.config.floatx()))
    output_shape = output.shape[1:]
    backward_layer = get_backward(layer)

    mask_output = Input(output_shape)
    input_ = Input(input_shape)
    output = backward_layer([mask_output, input_])
    model_backward = Model([mask_output, input_], output)
    for i in range(output_shape[-1]):
        compute_backward_layer(
            input_shape,
            model,
            model_backward,
            input_random=True,
            clip_min=clip_min,
            clip_max=clip_max,
            name=layer.__class__.__name__,
            index=i,
        )


@pytest.mark.parametrize(
    "keras_layer, input_shape",
    [
        (Norm(ord=2, axis=-1), (32, 4)),
    ],
)
def test_backward_unary_norm(keras_layer, input_shape):
    # _test_backward(keras_layer)
    if keras.backend.backend() == "jax":
        pytest.skip("Skipping test for JAX backend (not supported).")
    _test_backward(keras_layer, input_shape=input_shape)


@pytest.mark.parametrize(
    "keras_layer",
    [
        Abs(),
        Absolute(),
        Arccos(),
        Arcsinh(),
        Arctan(),
        Cos(),
        Cosh(),
        Expm1(),
        Floor(),
        Negative(),
        Prod(axis=-1, keepdims=True),
        Reciprocal(),
        Sign(),
        Sin(),
        Sinh(),
        Square(),
        Sort(axis=-1),
        Tan(),
        ExpandDims(axis=-2),
        Repeat(repeats=3, axis=-1),
        Roll(shift=2, axis=-1),
        # Trace(),
        Trunc(),
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
        (Tan(), -1.0, 1.0),
        # Trace(),
        # Transpose(axes=(0, 2, 1)),
        # Trunc(),
        # Var(axis=-1),
        # ZerosLike(),
    ],
)
def test_backward_unary_ops_clip(keras_layer, clip_min, clip_max):
    _test_backward(keras_layer, clip_min=clip_min, clip_max=clip_max)


@pytest.mark.parametrize(
    "keras_layer, input_shape",
    [
        (AMax(axis=-1, keepdims=False), (3, 2)),
        (AMax(axis=-1, keepdims=True), (3, 2)),
        (AMin(axis=-1, keepdims=False), (3, 2)),
        (AMin(axis=-1, keepdims=True), (3, 2)),
        (Var(axis=-1), (32, 4)),
        (Cumsum(axis=-1), (3, 4)),
        (Cumsum(axis=-2), (3, 4)),
        (Diagonal(axis1=1, axis2=2), (32, 4)),
        (SwapAxes(axis1=1, axis2=2), (32, 4)),
        (Transpose(axes=(0, 2, 1)), (32, 4)),
        (MoveAxis(2, 1), (3, 2)),
    ],
)
def test_backward_unary_ops_vector(keras_layer, input_shape):
    _test_backward(keras_layer, input_shape=input_shape)
