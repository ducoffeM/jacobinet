import keras
import keras.ops as K  # type: ignore
import numpy as np
from jacobinet.layers.layer import BackwardLinearLayer
from keras.layers import (  # type: ignore
    AveragePooling1D,
    Conv1DTranspose,
    Layer,
    ZeroPadding1D,
)
from keras.models import Sequential  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardAveragePooling1D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `AveragePooling1D` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the average pooling
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import AveragePooling1D
    from keras_custom.backward.layers import BackwardAveragePooling1D

    # Assume `average_pooling_layer` is a pre-defined AveragePooling1D layer
    backward_layer = BackwardAveragePooling1D(average_pooling_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: AveragePooling1D,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)

        if not self.layer.built:
            raise ValueError("layer {} is not built".format(layer.name))

        if self.layer.padding == "same":
            raise UserWarning(
                "Padding same with AveragePooling is not reperoductible in pure keras operations. See github issue https://github.com/mil-tokyo/webdnn/issues/694"
            )

        # average pooling is a depthwise convolution
        # we use convtranspose to invert the convolution of kernel ([1/n..1/n]..[1/n..1/n]) with n the pool size
        pool_size = list(layer.pool_size)
        layer_t = Conv1DTranspose(
            1,
            layer.pool_size,
            strides=layer.strides,
            padding=self.layer.padding,
            data_format=layer.data_format,
            use_bias=False,
            trainable=False,
        )
        kernel_ = np.ones(pool_size + [1, 1], dtype="float32") / np.prod(pool_size)
        layer_t.kernel = keras.Variable(kernel_.astype("float32"))
        layer_t.built = True

        # shape of transposed input
        # input_shape_t = list(layer_t(self.layer.output).shape[1:])
        input_shape_t = list(
            layer_t(K.ones((1,) + self.output_dim_wo_batch, dtype=keras.config.floatx())).shape[1:]
        )
        # input_shape = list(self.layer.input.shape[1:])
        input_shape = self.input_dim_wo_batch

        if layer.data_format == "channels_first":
            h_pad = input_shape[-1] - input_shape_t[-1]
        else:
            h_pad = input_shape[0] - input_shape_t[0]

        if h_pad:
            padding = (0, h_pad)
            self.model = Sequential(
                [
                    layer_t,
                    ZeroPadding1D(padding, data_format=self.layer.data_format),
                ]
            )
        else:
            self.model = Sequential([layer_t])
        # self.model(self.layer.output)
        self.model(K.ones((1,) + self.output_dim_wo_batch, dtype=keras.config.floatx()))
        self.model.trainable = False
        self.model.built = True

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


def get_backward_AveragePooling1D(layer: AveragePooling1D) -> Layer:
    """
    This function creates a `BackwardAveragePooling1D` layer based on a given `AveragePooling1D` layer. It provides
    a convenient way to obtain a backward approximation of the input `AveragePooling1D` layer, using the
    `BackwardAveragePooling1D` class to reverse the average pooling operation.

    ### Parameters:
    - `layer`: A Keras `AveragePooling1D` layer instance. The function uses this layer's configurations to set up the `BackwardAveragePooling1D` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardAveragePooling1D`, which acts as the reverse layer for the given `AveragePooling1D`.

    ### Example Usage:
    ```python
    from keras.layers import AveragePooling1D
    from keras_custom.backward import get_backward_AveragePooling1D

    # Assume `average_layer` is a pre-defined AveragePooling1D layer
    backward_layer = get_backward_AveragePooling1D(average_layer)
    output = backward_layer(input_tensor)
    """

    layer_backward = BackwardAveragePooling1D(layer)
    return layer_backward
