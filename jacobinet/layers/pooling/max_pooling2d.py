import numpy as np
import math
from keras.layers import Layer, Input, MaxPooling2D, Reshape, ZeroPadding2D
from keras.models import Sequential, Model
import keras.ops as K
from jacobinet.layers.utils import pooling_layer2D
from jacobinet.layers.utils import reshape_to_batch

from jacobinet.layers.layer import BackwardNonLinearLayer
from jacobinet.layers.reshaping import BackwardReshape
from jacobinet.layers.convolutional import BackwardDepthwiseConv2D
from jacobinet.layers.custom.prime import max_prime
from .utils_conv import get_conv_op
from keras import KerasTensor as Tensor

from .utils_max import ConstantPadding2D

from typing import List, Any

def get_pad_width(input_shape_wo_batch:List[int], pool_size:List[int], strides:List[int], data_format:str)->List[List[int]]:

    if data_format=="channels_first":
        input_shape_wh = input_shape_wo_batch[1:]
    else:
        input_shape_wh = input_shape_wo_batch[:2]

    # width
    # shape with valid
    output_shape_w_valid = math.floor((input_shape_wh[0] - pool_size[0]) / strides[0]) + 1 
    # shape with same
    output_shape_w_same = math.floor((input_shape_wh[0] - 1) / strides[0]) + 1

    output_shape_h_valid = math.floor((input_shape_wh[1] - pool_size[1]) / strides[1]) + 1 
    # shape with same
    output_shape_h_same = math.floor((input_shape_wh[1] - 1) / strides[1]) + 1

    w_pad:int = output_shape_w_same - output_shape_w_valid
    h_pad:int = output_shape_h_same - output_shape_h_valid
    # split in top, bottom, left, right
    pad_top:int = w_pad//2
    pad_bottom:int = pad_top + w_pad%2
    pad_left:int = h_pad//2
    pad_right:int = pad_left + h_pad%2

    pad_batch = [0,0]
    pad_channel = [0,0]
    pad_width = [pad_top, pad_bottom]
    pad_height = [pad_left, pad_right]

    padding = [pad_width, pad_height]

    if data_format=="channels_first":
        #pad_width = [pad_batch, pad_channel, pad_width, pad_height]
        pad_width = [pad_batch, pad_channel]+padding
    else:
        #pad_width = [pad_batch, pad_width, pad_height, pad_channel]
        pad_width = [pad_batch]+padding + [pad_channel]

    return pad_width, padding


class BackwardMaxPooling2D(BackwardNonLinearLayer):
    """
    This class implements a custom layer for backward pass of a `MaxPooling2D` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the max pooling
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import MaxPooling2D
    from keras_custom.backward.layers import BackwardMaxPooling2D

    # Assume `max_pooling_layer` is a pre-defined MaxPooling2D layer
    backward_layer = BackwardMaxPooling2D(max_pooling_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: MaxPooling2D,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)
        conv_op, _ = get_conv_op(self.layer, self.input_dim_wo_batch)
        self.conv_op = conv_op
        # if padding = same, we need to add the padding manually as it is not the same padding in convolution
        # in that case padding is constant with padding_valud -np.inf
        if self.layer.padding =='same':
            # compute pad_width
            self.pad_width, padding = get_pad_width(input_shape_wo_batch = self.input_dim_wo_batch, pool_size=self.layer.pool_size, strides=self.layer.strides, data_format=self.layer.data_format )
            

        x = Input(self.input_dim_wo_batch)
        
        if self.layer.padding=='same':
            self.padding_layer = ConstantPadding2D(const=-1e10, padding=padding, data_format=self.layer.data_format)
            x_0 = self.padding_layer(x)
        else:
            x_0 = x
        x_1 = self.conv_op(x_0)
        conv_out_shape_wo_batch = list(
            x_1.shape[1:]
        )

        if self.layer.data_format == "channels_first":

            in_channel = self.input_dim_wo_batch[0]
            image_shape = conv_out_shape_wo_batch[-2:]
            #image_shape = self.output_dim_wo_batch[-2:]
            self.target_shape = [in_channel, -1] + image_shape
            self.axis = 2
        else:
            in_channel = self.input_dim_wo_batch[-1]
            image_shape = conv_out_shape_wo_batch[:2]
            #image_shape = self.output_dim_wo_batch[:2]
            self.target_shape = image_shape + [in_channel, -1]
            self.axis = -1

        self.reshape_op = Reshape(self.target_shape)
        # compile a model to init shape
        #inner_model = Sequential([self.conv_op, self.reshape_op])
        x_2 = self.reshape_op(x_1)

        # init input/output of every layer by creating a model
        self.linear_block = Model(x, x_2)
        self.backward_reshape_op = BackwardReshape(layer=self.reshape_op)

        backward_conv2d = BackwardDepthwiseConv2D(layer=self.conv_op)

        # to do
        if self.layer.padding =='valid':
            self.backward_conv2d=backward_conv2d
        else:
            # do cropping !!!
            # input_shape_wo_batch = list(layer.input.shape[1:])
            input_shape_wo_batch = self.input_dim_wo_batch
            # input_shape_wo_batch_wo_pad = list(layer_backward(layer.output)[0].shape)
            input_shape_wo_batch_wo_pad = list(
            backward_conv2d(Input(self.output_dim_wo_batch))[0].shape
            )

            if layer.data_format == "channels_first":
                w_pad = input_shape_wo_batch[1] - input_shape_wo_batch_wo_pad[1]
                h_pad = input_shape_wo_batch[2] - input_shape_wo_batch_wo_pad[2]
            else:
                w_pad = input_shape_wo_batch[0] - input_shape_wo_batch_wo_pad[0]
                h_pad = input_shape_wo_batch[1] - input_shape_wo_batch_wo_pad[1]

            pad_layers = pooling_layer2D(w_pad, h_pad, layer.data_format)
            if len(pad_layers):
                self.backward_conv2d = Sequential([backward_conv2d] + pad_layers)
                # init
                self.backward_conv2d(Input(self.output_dim_wo_batch))

        self.linear_block_backward = Sequential([self.backward_reshape_op, backward_conv2d])

    def call(self, inputs, training=None, mask=None):

        layer_input = None

        gradient = inputs[0] # (batch, n_out..., C_W_out, H_out)
        layer_input = inputs[1] # (batch, C, W_in, H_in)

        reshape_tag, gradient_, n_out = reshape_to_batch(
            gradient, [1] + self.output_dim_wo_batch
        )
        # gradient_ (batch*prod(n_out), output_dim_wo_batch...)= (batch*prod(n_out), C, W_out, H_out) if channel_first

        input_max = self.linear_block(layer_input) # (batch, C, pool_size, W_out, H_out)
        backward_max = max_prime(input_max, axis=self.axis) # (batch, C, pool_size, W_out, H_out)
        # combine backward_max with gradient, we first need to reshape gradient_
        inner_input_dim_wo_batch = list(backward_max.shape[1:4]) # =(C, pool_size, W_out, H_out)
        if len(n_out):
            gradient_ = K.reshape(gradient_, [-1, np.prod(n_out)]+self.output_dim_wo_batch) #(batch, N_out, C, W_out, H_out)

        # n_out=[] gradient_.shape = (batch, N_out, C, W_out, H_out) else (batch, C, W_out, H_out)
        if len(n_out):
            if self.layer.data_format=="channel_first":
                gradient_ = K.expand_dims(gradient_, self.axis+1) #(batch, N_out, C, 1, W_out, H_out)
                backward_max = K.expand_dims(backward_max, 1) # (batch, 1, C, pool_size, W_out, H_out)
            else:
                # gradient_ (batch, N_out, W_out, H_out, C)
                # backward_max (batch, W_out, H_out, C, pool_size)
                gradient_ = K.expand_dims(gradient_, self.axis) #(batch, N_out, W_out, H_out, C, 1)
                backward_max = K.expand_dims(backward_max, 1) # (batch, 1, W_out, H_out, C, pool_size)  
        else:
            #gradient_ (batch, C, W_out, H_out)
            # backward_max (batch, C, pool_size, W_out, H_out)
            gradient_ = K.expand_dims(gradient_, self.axis) #(batch, C, 1, W_out, H_out)

        # element wise product to apply chain rule
        gradient_ = gradient_*backward_max # (batch, N_out, C, pool_size, W_out, H_out)
        # reshape 
        if len(n_out):
            gradient_ = K.reshape(gradient_, [-1]+inner_input_dim_wo_batch) # (batch*N_out, C, pool_size, W_out, H_out)

        # backward_reshape
        #backward_reshape = self.backward_reshape_op(gradient_) # (batch*N_out, C*pool_size, W_out, H_out)
        #output = self.backward_conv2d(backward_reshape) #(batch*N_out, C, W_in, H_in)
        output = self.linear_block_backward(gradient_) #(batch*N_out, C, W_in, H_in)

        # reshape_tag
        if reshape_tag:
            output = K.reshape(
                    output, [-1] + n_out + self.input_dim_wo_batch
                )

        return output


def get_backward_MaxPooling2D(layer: MaxPooling2D) -> Layer:
    """
    This function creates a `BackwardMaxPooling2D` layer based on a given `MaxPooling2D` layer. It provides
    a convenient way to obtain a backward approximation of the input `MaxPooling2D` layer, using the
    `BackwardMaxPooling2D` class to reverse the max pooling operation.

    ### Parameters:
    - `layer`: A Keras `MaxPooling2D` layer instance. The function uses this layer's configurations to set up the `BackwardMaxPooling2D` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardMaxPooling2D`, which acts as the reverse layer for the given `MaxPooling2D`.

    ### Example Usage:
    ```python
    from keras.layers import MaxPooling2D
    from keras_custom.backward import get_backward_MaxPooling2D

    # Assume `maxpool_layer` is a pre-defined MaxPooling2D layer
    backward_layer = get_backward_MaxPooling2D(maxpool_layer)
    output = backward_layer(input_tensor)
    """

    layer_backward = BackwardMaxPooling2D(layer)
    return layer_backward
