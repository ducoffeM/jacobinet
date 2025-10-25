from typing import Callable, List, Tuple, Union

import keras
import keras.ops as K  # type:ignore
import numpy as np
from jacobinet.utils import to_tuple
from keras import KerasTensor as Tensor
from keras.layers import (  # type:ignore
    Cropping1D,
    Cropping2D,
    Cropping3D,
    Input,
    Layer,
    Permute,
    ZeroPadding1D,
    ZeroPadding2D,
    ZeroPadding3D,
)


# compute output shape post convolution
def compute_output_pad(
    input_shape_wo_batch: Tuple[int, int, int],
    kernel_size: Tuple[int, int],
    strides: Tuple[int, int],
    padding: str,
    data_format: str,
) -> Tuple[int, int]:
    """
    Computes the padding required for the output shape after applying a convolution operation.

    Args:
        input_shape_wo_batch: The shape of the input (excluding batch size),
                                      in the form (channel, width, height) or (width, height, channel)
                                      depending on the data format.
        kernel_size: The size of the convolutional kernel (k_w, k_h).
        strides: The strides for the convolution (s_w, s_h).
        padding: Type of padding ("same" or "valid").
        data_format: Data format of the input, either "channels_first" or "channels_last".

    Returns:
        tuple: (w_pad, h_pad), representing the padding required for width and height.

    Notes:
        - If padding is "same", the function assumes no additional padding is needed (p = 0).
        - The function calculates the padding based on the kernel size, stride, and input shape.
    """
    if data_format == "channels_first":
        w, h = input_shape_wo_batch[1:]
    else:
        w, h = input_shape_wo_batch[:-1]
    k_w, k_h = kernel_size
    if padding == "same":
        p = 0
    s_w, s_h = strides

    w_pad = (w - k_w + 2 * p) / s_w + 1 - w
    h_pad = (h - k_h + 2 * p) / s_h + 1 - h
    return (int(w_pad), int(h_pad))


def pooling_layer2D(
    w_pad: int, h_pad: int, data_format: str
) -> List[Union[ZeroPadding2D, Cropping2D]]:
    """
    Determines the padding and cropping layers to apply based on the width and height padding.

    This function will return a list of 2D padding or cropping layers (e.g., `ZeroPadding2D`, `Cropping2D`)
    based on the given padding values for width and height. It handles different cases based on whether
    the padding values are positive or negative, and adjusts the layers accordingly.

    Args:
        w_pad: The width padding to apply. A positive value indicates padding,
                     and a negative value indicates cropping.
        h_pad: The height padding to apply. A positive value indicates padding,
                     and a negative value indicates cropping.
        data_format: The data format to use, either 'channels_first' or 'channels_last'.

    Returns:
        List[Union[ZeroPadding2D, Cropping2D]]: A list containing either `ZeroPadding2D` or `Cropping2D`
                                                layers depending on the padding and cropping requirements.
                                                If no padding or cropping is needed, an empty list is returned.

    Notes:
        - If both `w_pad` and `h_pad` are positive, `ZeroPadding2D` is used.
        - If both `w_pad` and `h_pad` are negative, `Cropping2D` is used.
        - If one of the dimensions is positive and the other is negative, both `ZeroPadding2D` and `Cropping2D`
          layers are applied to handle padding and cropping accordingly.
    """
    if w_pad or h_pad:
        # add padding
        if w_pad >= 0 and h_pad >= 0:
            padding = (
                (w_pad // 2, w_pad // 2 + w_pad % 2),
                (h_pad // 2, h_pad // 2 + h_pad % 2),
            )
            pad_layer = [ZeroPadding2D(padding, data_format=data_format)]
        elif w_pad <= 0 and h_pad <= 0:
            w_pad *= -1
            h_pad *= -1
            # padding = ((0, -w_pad), (0, -h_pad))
            cropping = (
                (w_pad // 2, w_pad // 2 + w_pad % 2),
                (h_pad // 2, h_pad // 2 + h_pad % 2),
            )
            pad_layer = [Cropping2D(cropping, data_format=data_format)]
        elif w_pad > 0 and h_pad < 0:
            h_pad *= -1
            padding = ((w_pad // 2, w_pad // 2 + w_pad % 2), (0, 0))
            cropping = ((0, 0), (h_pad // 2, h_pad // 2 + h_pad % 2))
            pad_layer = [
                ZeroPadding2D(padding, data_format=data_format),
                Cropping2D(cropping, data_format=data_format),
            ]
        else:
            w_pad *= -1
            padding = ((0, 0), (h_pad // 2, h_pad // 2 + h_pad % 2))
            cropping = ((w_pad // 2, w_pad // 2 + w_pad % 2), (0, 0))
            pad_layer = [
                ZeroPadding2D(padding, data_format=data_format),
                Cropping2D(cropping, data_format=data_format),
            ]
        return pad_layer
    return []


def pooling_layer1D(w_pad: int, data_format: str) -> List[Union[ZeroPadding1D, Cropping1D]]:
    """
    Determines the padding and cropping layers to apply based on the width padding.

    This function will return a list of 1D padding or cropping layers (e.g., `ZeroPadding1D`, `Cropping1D`)
    based on the given padding values for width. It handles different cases based on whether
    the padding values are positive or negative, and adjusts the layers accordingly.

    Args:
        w_pad: The width padding to apply. A positive value indicates padding,
                     and a negative value indicates cropping.
        data_format: The data format to use, either 'channels_first' or 'channels_last'.

    Returns:
        List[Union[ZeroPadding1D, Cropping1D]]: A list containing either `ZeroPadding1D` or `Cropping1D`
                                                layers depending on the padding and cropping requirements.
                                                If no padding or cropping is needed, an empty list is returned.

    """
    if w_pad:
        # add padding
        if w_pad >= 0:
            padding = (w_pad // 2, w_pad // 2 + w_pad % 2)
            pad_layer = [ZeroPadding1D(padding, data_format=data_format)]
        else:
            w_pad *= -1
            # padding = ((0, -w_pad), (0, -h_pad))
            cropping = (w_pad // 2, w_pad // 2 + w_pad % 2)
            if data_format == "channels_first":
                # we need to permute the layer to apply Cropping on the right dimensions
                perm_layer = Permute((2, 1))
                pad_layer = [perm_layer, Cropping1D(cropping), perm_layer]
            else:
                pad_layer = [Cropping1D(cropping)]
        return pad_layer
    return []


def pooling_layer3D(
    d_pad: int, w_pad: int, h_pad: int, data_format: str
) -> List[Union[ZeroPadding3D, Cropping3D]]:
    """
    Determines the padding and cropping layers to apply based on the width, height and depth padding.

    This function will return a list of 3D padding or cropping layers (e.g., `ZeroPadding3D`, `Cropping3D`)
    based on the given padding values for width and height. It handles different cases based on whether
    the padding values are positive or negative, and adjusts the layers accordingly.

    Args:
        d_pad: The depth padding to apply. A positive value indicates padding,
                     and a negative value indicates cropping.
        w_pad: The width padding to apply. A positive value indicates padding,
                     and a negative value indicates cropping.
        h_pad: The height padding to apply. A positive value indicates padding,
                     and a negative value indicates cropping.
        data_format: The data format to use, either 'channels_first' or 'channels_last'.

    Returns:
        List[Union[ZeroPadding3D, Cropping3D]]: A list containing either `ZeroPadding3D` or `Cropping3D`
                                                layers depending on the padding and cropping requirements.
                                                If no padding or cropping is needed, an empty list is returned.

    """
    padding: Union[
        Tuple[Tuple[int, int], Tuple[int, int]],  # 2 times
        Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],  # 3 times
        Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]],  # 4 times
    ]
    cropping: Union[
        Tuple[Tuple[int, int], Tuple[int, int]],  # 2 times
        Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],  # 3 times
        Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]],  # 4 times
    ]
    if d_pad or w_pad or h_pad:
        # add padding
        if d_pad >= 0 and w_pad >= 0 and h_pad >= 0:
            padding = (
                (d_pad // 2, d_pad // 2 + d_pad % 2),
                (w_pad // 2, w_pad // 2 + w_pad % 2),
                (h_pad // 2, h_pad // 2 + h_pad % 2),
            )
            pad_layer = [ZeroPadding3D(padding, data_format=data_format)]
        elif d_pad <= 0 and w_pad <= 0 and h_pad <= 0:
            d_pad *= -1
            w_pad *= -1
            h_pad *= -1
            # padding = ((0, -w_pad), (0, -h_pad))
            cropping = (
                (d_pad // 2, d_pad // 2 + d_pad % 2),
                (w_pad // 2, w_pad // 2 + w_pad % 2),
                (h_pad // 2, h_pad // 2 + h_pad % 2),
            )
            pad_layer = [Cropping3D(cropping, data_format=data_format)]
        elif w_pad > 0 and h_pad < 0:
            h_pad *= -1
            padding = ((w_pad // 2, w_pad // 2 + w_pad % 2), (0, 0))
            cropping = ((0, 0), (h_pad // 2, h_pad // 2 + h_pad % 2))
            pad_layer = [
                ZeroPadding2D(padding, data_format=data_format),
                Cropping2D(cropping, data_format=data_format),
            ]
        else:
            w_pad *= -1
            padding = ((0, 0), (h_pad // 2, h_pad // 2 + h_pad % 2))
            cropping = ((w_pad // 2, w_pad // 2 + w_pad % 2), (0, 0))
            pad_layer = [
                ZeroPadding2D(padding, data_format=data_format),
                Cropping2D(cropping, data_format=data_format),
            ]
        return pad_layer
    else:
        return []


def call_backward_depthwise2d(
    inputs: Tensor,
    layer: Layer,
    op_reshape: Callable,
    op_split: Callable,
    inner_models: List[Callable],
    axis: int,
    axis_c: int,
    c_in: int,
    use_bias: bool,
) -> Tensor:
    """
    Performs the backward pass for a depthwise 2D convolution layer.

    This function handles the removal of bias (if necessary), reshapes the input tensor,
    splits it along the specified axis, applies individual convolution operations to
    each split, and finally concatenates the results along the specified axis.

    Args:
        inputs: The input tensor to the layer. It is expected to have the shape
                          (batch, height, width, channels) or (batch, channels, height, width)
                          depending on the data format.
        layer: The layer object that provides the bias (if applicable) and other
                       configurations like data format.
        op_reshape: A function used to reshape the input tensor before processing.
        op_split: A function used to split the reshaped tensor along the specified axis.
        inner_models: A list of functions or models that process each split of the tensor.
        axis: The axis along which the results of the depthwise convolutions are concatenated.
        axis_c: The axis along which the input tensor is split.
        c_in: The number of input channels.
        use_bias: A flag indicating whether to use bias in the layer computation.

    Returns:
        The concatenated result of the processed splits after applying the inner models
                (depthwise convolutions).

    Notes:
        - If `use_bias` is true and the layer has a bias, it is subtracted from the input tensor
          before any further operations.
        - The function expects the data format to be either "channels_first" or "channels_last"
          and handles reshaping and splitting accordingly.
        - The depthwise convolutions are applied separately to each split of the input, and
          the results are concatenated along the specified axis.

    """

    # remove bias if needed
    if hasattr(layer, "use_bias") and layer.use_bias and use_bias:
        if layer.data_format == "channels_first":
            inputs = inputs - layer.bias[None, :, None, None]  # (batch, d_m*c_in, w_out, h_out)
        else:
            inputs = inputs - layer.bias[None, None, None, :]  # (batch, w_out, h_out, d_m*c_in)

    outputs = op_reshape(inputs)  # (batch, d_m, c_in, w_out, h_out) if data_format=channel_first

    # if self.layer.use_bias and self.use_bias:

    split_outputs = K.split(outputs, c_in, axis=axis_c)  # [(batch, d_m, 1, w_out, h_out)]
    split_outputs = [
        op_split(s_o_i) for s_o_i in split_outputs
    ]  # [(batch_size, d_m, w_out, h_out)]

    conv_outputs = [
        inner_models[i](s_o_i) for (i, s_o_i) in enumerate(split_outputs)
    ]  # [(batch_size, 1, w_in, h_in)]
    return K.concatenate(conv_outputs, axis=axis)  # (batch_size, c_in, w_in, h_in)


def reshape_to_batch(
    input_tensor: Tensor, layer_output_shape: List[int]
) -> Tuple[bool, Tensor, List[int]]:
    """
    Reshapes the input tensor to batch format if its shape does not match the expected layer output shape.

    If the shape of the input tensor is not compatible with the expected output shape (i.e.,
    the tensor's shape includes additional dimensions before the batch dimension), the function
    reshapes the tensor such that all non-batch dimensions are merged into one batch dimension.
    The function also returns the original dimensions that were merged.

    Args:
        input_tensor: The input tensor to reshape.
        layer_output_shape: The expected shape of the output from the layer,
                                        including the batch size as the first dimension.

    Returns:
        Tuple[bool, Tensor, List[int]]:
            - bool: Indicates if the reshaping occurred (`True` if reshaped, `False` otherwise).
            - Tensor: The reshaped input tensor if reshaping occurred, otherwise the original tensor.
            - List[int]: The original dimensions (excluding the batch dimension) that were merged.

    Notes:
        - This function is useful for handling cases where the input tensor's shape includes
          additional dimensions before the batch dimension, and reshaping is necessary to
          maintain compatibility with the expected output shape.
        - The reshaped tensor will have the batch size combined with the rest of the dimensions
          into one dimension, followed by the layer's output shape.
    """
    if len(input_tensor.shape) != len(layer_output_shape):
        # backward is not in a diagonal format, current shape is [batch_size]+n_out+layer_output_shape
        # we reshape it to [batch_size*np.prod(n_out_)]+layer_output_shape
        n_out = list(input_tensor.shape[: -len(layer_output_shape[1:])][1:])
        return (
            True,
            K.reshape(input_tensor, (-1,) + to_tuple(layer_output_shape[1:])),
            n_out,
        )
    else:
        return False, input_tensor, []


# source: decomon/keras_utils.py
# author: nolwen huet
def share_weights_and_build(
    original_layer: Layer,
    new_layer: Layer,
    weight_names: list[str],
    input_shape_wo_batch: list[int],
) -> None:
    """Share the weights specidifed by names of an already built layer to another unbuilt layer.

    We assume that each weight is also an original_laer's attribute whose name is the weight name.

    Args:
        original_layer: the layer used to share the weights
        new_layer: the new layer which will be buit and will share the weights of the original layer
        weight_names: names of the weights to share

    Returns:

    """
    # Check the original_layer is built and the new_layer is not built
    if not original_layer.built:
        raise ValueError("The original layer must already be built for sharing its weights.")
    if new_layer.built:
        raise ValueError("The new layer must not be built to get the weights of the original layer")

    # store the weights as a new_layer variable before build (ie before the lock)
    for w_name in weight_names:
        w = getattr(original_layer, w_name)
        try:
            setattr(new_layer, w_name, w)
        except AttributeError:
            # manage hidden weights introduced for LoRA https://github.com/keras-team/keras/pull/18942
            w_name = f"_{w_name}"
            w = getattr(original_layer, w_name)
            setattr(new_layer, w_name, w)

    # build the layer
    new_layer(Input(input_shape_wo_batch))
    # overwrite the newly generated weights and untrack them
    for w_name in weight_names:
        w = getattr(original_layer, w_name)
        w_to_drop = getattr(new_layer, w_name)
        try:
            setattr(new_layer, w_name, w)
        except AttributeError:
            # manage hidden weights introduced for LoRA https://github.com/keras-team/keras/pull/18942
            w_name = f"_{w_name}"
            w = getattr(original_layer, w_name)
            w_to_drop = getattr(new_layer, w_name)
            setattr(new_layer, w_name, w)
        # untrack the not used anymore weight
        new_layer._tracker.untrack(w_to_drop)


def get_W(layer: Layer, input_dim: int) -> Tensor:
    """
    Numerically estimate the weight matrix W of a linear layer
    by probing it with one-hot vectors.

    Args:
        layer: A linear layer (Dense-like) with call(x) = W x + b
        input_dim: Dimensionality of the input space.

    Returns:
        W: The reconstructed weight matrix of shape (output_dim, input_dim).
    """
    # Compute bias first
    b = get_b(layer, input_dim)  # shape: (output_dim,)

    # Build one-hot input basis
    n: int = np.prod(input_dim)
    eye = K.reshape(
        K.eye(n, dtype=keras.backend.floatx()), (n,) + input_dim
    )  # (input_dim, input_dim)
    outputs = layer(eye)  # (n, output_dim)

    # Subtract bias
    W = outputs - b[None]
    return W  # (n, output_dim)


def get_b(layer: Layer, input_dim: int) -> Tensor:
    """
    Estimate the bias vector b = layer(0)
    """
    zero_input = K.zeros((1,) + input_dim)
    return layer(zero_input)[0]
