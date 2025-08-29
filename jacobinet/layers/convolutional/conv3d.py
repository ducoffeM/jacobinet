from typing import Any

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.core.activations import BackwardActivation
from jacobinet.layers.layer import BackwardLinearLayer, BackwardWithActivation
from jacobinet.layers.utils import pooling_layer3D
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras.layers import Conv3D, Conv3DTranspose, Input  # type: ignore
from keras.models import Sequential  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardConv3D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Conv3D` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Conv3D
    from keras_custom.backward.layers import BackwardConv3D

    # Assume `conv_layer` is a pre-defined Conv3D layer
    backward_layer = BackwardConv3D(conv_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Conv3D,
        **kwargs: Any,
    ):
        super().__init__(layer=layer, **kwargs)
        dico_conv = layer.get_config()
        dico_conv.pop("groups")
        # input_shape = list(layer.input.shape[1:])
        input_shape = self.input_dim_wo_batch
        # update filters to match input, pay attention to data_format
        if layer.data_format == "channels_first":  # better to use enum than raw str
            dico_conv["filters"] = input_shape[0]
        else:
            dico_conv["filters"] = input_shape[-1]

        dico_conv["use_bias"] = False
        dico_conv["padding"] = "valid"

        layer_backward = Conv3DTranspose.from_config(dico_conv)
        layer_backward.kernel = layer.kernel

        layer_backward.built = True

        # input_shape_wo_batch = list(layer.input.shape[1:])
        input_shape_wo_batch = self.input_dim_wo_batch
        # input_shape_wo_batch_wo_pad = list(layer_backward(layer.output)[0].shape)
        input_shape_wo_batch_wo_pad = list(
            layer_backward(K.ones([1] + self.output_dim_wo_batch, dtype=keras.config.floatx()))[
                0
            ].shape
        )

        if layer.data_format == "channels_first":
            d_pad = input_shape_wo_batch[1] - input_shape_wo_batch_wo_pad[1]
            w_pad = input_shape_wo_batch[2] - input_shape_wo_batch_wo_pad[2]
            h_pad = input_shape_wo_batch[3] - input_shape_wo_batch_wo_pad[3]
        else:
            d_pad = input_shape_wo_batch[0] - input_shape_wo_batch_wo_pad[0]
            w_pad = input_shape_wo_batch[1] - input_shape_wo_batch_wo_pad[1]
            h_pad = input_shape_wo_batch[2] - input_shape_wo_batch_wo_pad[2]

        pad_layers = pooling_layer3D(d_pad, w_pad, h_pad, layer.data_format)
        if len(pad_layers):
            layer_backward = Sequential([layer_backward] + pad_layers)
            layer_backward(Input(self.output_dim_wo_batch))
            # _ = layer_backward([1]+self.output_dim_wo_batch)
        self.layer_backward = layer_backward


@keras.saving.register_keras_serializable()
class BackwardConv3DWithActivation(BackwardWithActivation):
    """
    This class implements a custom layer for backward pass of a `Conv3D` layer in Keras with a non linear activation function.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Conv3D
    from keras_custom.backward.layers import BackwardConv2DWithActivation

    # Assume `conv_layer` is a pre-defined Conv3D layer with an activation function
    backward_layer = BackwardConv3DWithActivation(conv_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Conv3D,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            backward_linear=BackwardConv3D,
            backward_activation=BackwardActivation,
            **kwargs,
        )


def get_backward_Conv3D(layer: Conv3D) -> Layer:
    """
    This function creates a `BackwardConv3D` layer based on a given `Conv3D` layer. It provides
    a convenient way to obtain the backward pass of the input `Conv3D` layer, using the
    `BackwardConv3D` class to reverse the convolution operation.

    ### Parameters:
    - `layer`: A Keras `Conv3D` layer instance. The function uses this layer's configurations to set up the `BackwardConv3D` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardConv3D`, which acts as the reverse layer for the given `Conv3D`.

    ### Example Usage:
    ```python
    from keras.layers import Conv3D
    from keras_custom.backward import get_backward_Conv3D

    # Assume `conv_layer` is a pre-defined Conv3D layer
    backward_layer = get_backward_Conv3D(conv_layer)
    output = backward_layer(input_tensor)
    """
    if layer.get_config()["activation"] == "linear":
        return BackwardConv3D(layer)
    else:
        return BackwardConv3DWithActivation(layer)
