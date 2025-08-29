import math
import warnings

# typing
from typing import List

import keras
import keras.ops as K  # type: ignore
import numpy as np
import numpy.typing as npt
from jacobinet.layers.layer import BackwardLinearLayer
from jacobinet.layers.utils import pooling_layer2D
from keras.layers import AveragePooling2D, Conv2DTranspose, Input, Layer  # type: ignore
from keras.models import Sequential  # type: ignore

ArrayLike = npt.ArrayLike
Tensor = keras.KerasTensor


@keras.saving.register_keras_serializable()
class CroppingReflect2D(keras.layers.Layer):
    """Custom Keras Layer that applies the backward pass for the padding of AveragePooling2D with padding same
    This operation operates a non standard pooling function (not ZeroPadding), but duplicate the last lines and cols to get the right shape
    """

    def __init__(
        self,
        input_dim_wo_batch: List[int],
        pad_top: int,
        pad_bottom: int,
        pad_left=int,
        pad_right=int,
        data_format: str = keras.backend.image_data_format(),
        **kwargs,
    ):
        """ """
        super(CroppingReflect2D, self).__init__(**kwargs)

        self.pad_top = pad_top
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.input_dim_wo_batch = input_dim_wo_batch
        self.data_format = data_format

    def call(self, inputs, training=None, mask=None):
        if self.data_format == "channels_last":
            axis_w = 1
            axis_h = 2
        else:
            axis_w = 2
            axis_h = 3
        # input (batch, C, H+pad_top+pad_bottom, W + pad_left+pad_right)
        W_pad, H_pad = (
            self.input_dim_wo_batch[axis_w - 1],
            self.input_dim_wo_batch[axis_h - 1],
        )

        W = W_pad - self.pad_top - self.pad_bottom
        H = H_pad - self.pad_left - self.pad_right

        if self.pad_top:
            # [(batch, C, pad_top, H_pad), (batch, C, 1, H_pad), (batch, C, W - 1 + pad_bottom, H_pad]
            input_top, input_first_row, inputs = K.split(
                inputs, [self.pad_top, 1 + self.pad_top], axis=axis_w
            )
            input_first_row = input_first_row + K.sum(input_top, axis=axis_w, keepdims=True)
            # then concat
            inputs = K.concatenate(
                [input_first_row, inputs], axis=axis_w
            )  # (batch, C, W+pad_bottom, H_pad)

        if self.pad_bottom:
            # [(batch, C, W, H_pad), (batch, C, 1, H_pad), (batch, C, pad_bottom, H_pad]
            inputs, input_last_row, input_bottom = K.split(inputs, [W - 1, W], axis=axis_w)
            input_last_row = input_last_row + K.sum(input_bottom, axis=axis_w, keepdims=True)
            # then concat
            inputs = K.concatenate([inputs, input_last_row], axis=axis_w)  # (batch, C, W, H_pad)

        if self.pad_left:
            # [(batch, C, W, pad_left), (batch, C, W, 1), (batch, C, W, H-1)]
            input_left, input_first_col, inputs = K.split(
                inputs, [self.pad_left, 1 + self.pad_left], axis=axis_h
            )
            input_first_col = input_first_col + K.sum(input_left, axis=axis_h, keepdims=True)
            # then concat
            inputs = K.concatenate(
                [input_first_col, inputs], axis=axis_h
            )  # (batch, C, W, H+pad_right)

        if self.pad_right:
            # [(batch, C, W, H_pad), (batch, C, 1, H_pad), (batch, C, pad_bottom, H_pad]
            inputs, input_last_col, input_right = K.split(inputs, [H - 1, H], axis=axis_h)
            input_last_col = input_last_col + K.sum(input_right, axis=axis_h, keepdims=True)
            # then concat
            inputs = K.concatenate([inputs, input_last_col], axis=axis_h)  # (batch, C, W, H)

        return inputs  # (batch, C, W, H)

    def compute_output_shape(self, input_shape):
        output_shape = (1,) + tuple(self.input_dim_wo_batch)
        return output_shape

    def get_config(self):
        config = super().get_config()
        config["pad_top"] = self.pad_top
        config["pad_bottom"] = self.pad_bottom
        config["pad_left"] = self.pad_left
        config["pad_right"] = self.pad_right
        config["input_dim_wo_batch"] = self.input_dim_wo_batch
        config["data_format"] = self.data_format
        return config

    @classmethod
    def from_config(cls, config):
        pad_top_config = config.pop("pad_top")
        pad_top = keras.saving.deserialize_keras_object(pad_top_config)

        pad_bottom_config = config.pop("pad_bottom")
        pad_bottom = keras.saving.deserialize_keras_object(pad_bottom_config)

        pad_left_config = config.pop("pad_left")
        pad_left = keras.saving.deserialize_keras_object(pad_left_config)

        pad_right_config = config.pop("pad_right")
        pad_right = keras.saving.deserialize_keras_object(pad_right_config)

        data_format_config = config.pop("data_format")
        data_format = keras.saving.deserialize_keras_object(data_format_config)

        input_dim_wo_batch_config = config.pop("input_dim_wo_batch")
        input_dim_wo_batch = keras.saving.deserialize_keras_object(input_dim_wo_batch_config)

        return cls(
            input_dim_wo_batch=input_dim_wo_batch,
            data_format=data_format,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            pad_left=pad_left,
            pad_right=pad_right,
            **config,
        )


@keras.saving.register_keras_serializable()
class BackwardAveragePooling2D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `AveragePooling2D` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the average pooling
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import AveragePooling2D
    from keras_custom.backward.layers import BackwardAveragePooling2D

    # Assume `average_pooling_layer` is a pre-defined AveragePooling2D layer
    backward_layer = BackwardAveragePooling2D(average_pooling_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: AveragePooling2D,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)

        if not self.layer.built:
            raise ValueError("layer {} is not built".format(layer.name))

        if self.layer.padding == "same":
            warnings.warn(
                "Padding same with AveragePooling is not reperoductible in pure keras operations. See github issue https://github.com/mil-tokyo/webdnn/issues/694"
            )

        # average pooling is a depthwise convolution
        # we use convtranspose to invert the convolution of kernel ([1/n..1/n]..[1/n..1/n]) with n the pool size
        pool_size = list(layer.pool_size)
        layer_t = Conv2DTranspose(
            1,
            layer.pool_size,
            strides=layer.strides,
            padding="valid",
            data_format=layer.data_format,
            use_bias=False,
            trainable=False,
        )
        kernel_ = np.ones(pool_size + [1, 1], dtype="float32") / np.prod(pool_size)
        layer_t.kernel = keras.Variable(kernel_.astype("float32"))
        layer_t.built = True

        if self.layer.data_format == "channels_last":
            output_dim_wo_batch_c_1 = self.output_dim_wo_batch[:-1] + (1,)
        else:
            output_dim_wo_batch_c_1 = (1,) + self.output_dim_wo_batch[1:]

        # shape of transposed input
        input_shape_t = list(
            layer_t(K.ones((1,) + output_dim_wo_batch_c_1, dtype=keras.config.floatx())).shape[1:]
        )

        if self.layer.padding == "valid":
            input_shape = self.input_dim_wo_batch
            if layer.data_format == "channels_first":
                w_pad = input_shape[-2] - input_shape_t[-2]
                h_pad = input_shape[-1] - input_shape_t[-1]
            else:
                w_pad = input_shape[0] - input_shape_t[0]
                h_pad = input_shape[1] - input_shape_t[1]

            pad_layers = pooling_layer2D(w_pad, h_pad, self.layer.data_format)

        else:
            if self.layer.data_format == "channels_first":
                input_shape_wh = list(self.input_dim_wo_batch[1:])
            else:
                input_shape_wh = list(self.input_dim_wo_batch[:2])
            # width
            # shape with valid
            output_shape_w_valid = (
                math.floor((input_shape_wh[0] - self.layer.pool_size[0]) / self.layer.strides[0])
                + 1
            )
            # shape with same
            output_shape_w_same = math.floor((input_shape_wh[0] - 1) / self.layer.strides[0]) + 1

            output_shape_h_valid = (
                math.floor((input_shape_wh[1] - self.layer.pool_size[1]) / self.layer.strides[1])
                + 1
            )
            # shape with same
            output_shape_h_same = math.floor((input_shape_wh[1] - 1) / self.layer.strides[1]) + 1

            w_pad: int = output_shape_w_same - output_shape_w_valid
            h_pad: int = output_shape_h_same - output_shape_h_valid
            # split in top, bottom, left, right
            pad_top: int = w_pad // 2
            pad_bottom: int = pad_top + w_pad % 2
            pad_left: int = h_pad // 2
            pad_right: int = pad_left + h_pad % 2

            inner_input_dim_wo_batch = [
                input_shape_wh[0] + pad_top + pad_bottom,
                input_shape_wh[1] + pad_left + pad_right,
            ]
            if self.layer.data_format == "channels_first":
                inner_input_dim_wo_batch = [self.input_dim_wo_batch[0]] + inner_input_dim_wo_batch
            else:
                inner_input_dim_wo_batch = inner_input_dim_wo_batch + [self.input_dim_wo_batch[0]]

            pad_layer = CroppingReflect2D(
                input_dim_wo_batch=inner_input_dim_wo_batch,
                pad_top=pad_top,
                pad_bottom=pad_bottom,
                pad_left=pad_left,
                pad_right=pad_right,
                data_format="channels_first",
            )
            pad_layers = [pad_layer]

        if len(pad_layers):
            self.model = Sequential([layer_t] + pad_layers)
        else:
            self.model = layer_t
        # self.model(self.layer.output)

        self.model(Input(output_dim_wo_batch_c_1))
        self.model.trainable = False
        self.model.built = True

    def compute_output_shape(self, input_shape):
        return self.layer.input.shape

    def call_on_reshaped_gradient(self, gradient, input=None, training=None, mask=None):
        # inputs (batch, channel_out, w_out, h_out)
        if self.layer.data_format == "channels_first":
            channel_out = gradient.shape[1]
            axis = 1
        else:
            channel_out = gradient.shape[-1]
            axis = -1

        split_inputs = K.split(gradient, channel_out, axis)
        # apply conv transpose on every of them
        outputs = K.concatenate([self.model(input_i) for input_i in split_inputs], axis)
        return outputs


def get_backward_AveragePooling2D(layer: AveragePooling2D) -> Layer:
    """
    This function creates a `BackwardAveragePooling2D` layer based on a given `AveragePooling2D` layer. It provides
    a convenient way to obtain a backward approximation of the input `AveragePooling2D` layer, using the
    `BackwardAveragePooling2D` class to reverse the average pooling operation.

    ### Parameters:
    - `layer`: A Keras `AveragePooling2D` layer instance. The function uses this layer's configurations to set up the `BackwardAveragePooling2D` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardAveragePooling2D`, which acts as the reverse layer for the given `AveragePooling2D`.

    ### Example Usage:
    ```python
    from keras.layers import AveragePooling2D
    from keras_custom.backward import get_backward_AveragePooling2D

    # Assume `average_layer` is a pre-defined AveragePooling2D layer
    backward_layer = get_backward_AveragePooling2D(average_layer)
    output = backward_layer(input_tensor)
    """

    layer_backward = BackwardAveragePooling2D(layer)
    return layer_backward
