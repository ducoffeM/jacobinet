import math
from typing import Any, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras_custom.layers.numpy import Log, Log1p, Log2, Log10  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardLog(BackwardNonLinearLayer):
    """
    This function creates a BackwardLog layer based on a given Log layer.
    It provides a convenient way to obtain the backward pass of the input Log layer,
    using the BackwardLog.

    ### Parameters:
    - layer: A Keras Log layer instance.
      The function uses this layer's configurations to set up the BackwardLog layer.

    ### Returns:
    - layer_backward: An instance of BackwardLog, which acts as the reverse layer for the given Log.

    ### Example Usage:
    ```python
    from keras.layers import Log
    from keras_custom.backward import get_backward_Log

    # Assume `log_layer` is a pre-defined Log layer
    backward_layer = get_backward_Log(log_layer)
    output = backward_layer(input_tensor)
    ```
    """

    def __init__(
        self,
        layer: Log,
        **kwargs: Any,
    ):
        super().__init__(layer=layer, **kwargs)

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # gradient of log(x) w.r.t x is 1/x
        return gradient / input


def get_backward_Log(layer: Log) -> Layer:
    """
    This function creates a `BackwardLog` layer based on a given `Log` layer.
    It provides a convenient way to obtain the backward pass of the input `Log` layer,
    using the `BackwardLog`.

    ### Parameters:
    - `layer`: A Keras `Log` layer instance.
      The function uses this layer's configurations to set up the `BackwardLog` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardLog`, which acts as the reverse layer for the given `Log`.

    ### Example Usage:
    ```python
    from keras.layers import Log
    from keras_custom.backward import get_backward_Log

    # Assume `log_layer` is a pre-defined Log layer
    backward_layer = get_backward_Log(log_layer)
    output = backward_layer(input_tensor)
    ```
    """
    return BackwardLog(layer)


@keras.saving.register_keras_serializable()
class BackwardLog10(BackwardNonLinearLayer):
    """
    This function creates a BackwardLog10 layer based on a given Log10 layer.
    It provides a convenient way to obtain the backward pass of the input Log10 layer,
    using the BackwardLog10.

    ### Parameters:
    - layer: A Keras Log10 layer instance.
      The function uses this layer's configurations to set up the BackwardLog10 layer.

    ### Returns:
    - layer_backward: An instance of BackwardLog10, which acts as the reverse layer for the given Log10.

    ### Example Usage:
    ```python
    from keras.layers import Log10
    from keras_custom.backward import get_backward_Log10

    # Assume `log10_layer` is a pre-defined Log10 layer
    backward_layer = get_backward_Log10(log10_layer)
    output = backward_layer(input_tensor)
    ```
    """

    def __init__(
        self,
        layer: Log10,
        **kwargs: Any,
    ):
        super().__init__(layer=layer, **kwargs)
        self._ln_10 = math.log(10)  # constant ln(10)

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # gradient of log10(x) = 1 / (x * ln(10))
        return gradient / (input * self._ln_10)


def get_backward_Log10(layer: Log10) -> Layer:
    """
    This function creates a `BackwardLog10` layer based on a given `Log10` layer.
    It provides a convenient way to obtain the backward pass of the input `Log10` layer,
    using the `BackwardLog10`.

    ### Parameters:
    - `layer`: A Keras `Log10` layer instance.
      The function uses this layer's configurations to set up the `BackwardLog10` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardLog10`, which acts as the reverse layer for the given `Log10`.

    ### Example Usage:
    ```python
    from keras.layers import Log10
    from keras_custom.backward import get_backward_Log10

    # Assume `log10_layer` is a pre-defined Log10 layer
    backward_layer = get_backward_Log10(log10_layer)
    output = backward_layer(input_tensor)
    ```
    """
    return BackwardLog10(layer)


@keras.saving.register_keras_serializable()
class BackwardLog1p(BackwardNonLinearLayer):
    """
    This function creates a BackwardLog1p layer based on a given Log1p layer.
    It provides a convenient way to obtain the backward pass of the input Log1p layer,
    using the BackwardLog1p.

    ### Parameters:
    - layer: A Keras Log1p layer instance.
      The function uses this layer's configurations to set up the BackwardLog1p layer.

    ### Returns:
    - layer_backward: An instance of BackwardLog1p, which acts as the reverse layer for the given Log1p.

    ### Example Usage:
    ```python
    from keras.layers import Log1p
    from keras_custom.backward import get_backward_Log1p

    # Assume `log1p_layer` is a pre-defined Log1p layer
    backward_layer = get_backward_Log1p(log1p_layer)
    output = backward_layer(input_tensor)
    ```
    """

    def __init__(
        self,
        layer: Log1p,
        **kwargs: Any,
    ):
        super().__init__(layer=layer, **kwargs)

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # gradient of log1p(x) = 1 / (1 + x)
        return gradient / (1 + input)


def get_backward_Log1p(layer: Log1p) -> Layer:
    """
    This function creates a `BackwardLog1p` layer based on a given `Log1p` layer.
    It provides a convenient way to obtain the backward pass of the input `Log1p` layer,
    using the `BackwardLog1p`.

    ### Parameters:
    - `layer`: A Keras `Log1p` layer instance.
      The function uses this layer's configurations to set up the `BackwardLog1p` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardLog1p`, which acts as the reverse layer for the given `Log1p`.

    ### Example Usage:
    ```python
    from keras.layers import Log1p
    from keras_custom.backward import get_backward_Log1p

    # Assume `log1p_layer` is a pre-defined Log1p layer
    backward_layer = get_backward_Log1p(log1p_layer)
    output = backward_layer(input_tensor)
    ```
    """
    return BackwardLog1p(layer)


@keras.saving.register_keras_serializable()
class BackwardLog2(BackwardNonLinearLayer):
    """
    This function creates a BackwardLog2 layer based on a given Log2 layer.
    It provides a convenient way to obtain the backward pass of the input Log2 layer,
    using the BackwardLog2.

    ### Parameters:
    - layer: A Keras Log2 layer instance.
      The function uses this layer's configurations to set up the BackwardLog2 layer.

    ### Returns:
    - layer_backward: An instance of BackwardLog2, which acts as the reverse layer for the given Log2.

    ### Example Usage:
    ```python
    from keras.layers import Log2
    from keras_custom.backward import get_backward_Log2

    # Assume `log2_layer` is a pre-defined Log2 layer
    backward_layer = get_backward_Log2(log2_layer)
    output = backward_layer(input_tensor)
    ```
    """

    def __init__(
        self,
        layer: Log2,
        **kwargs: Any,
    ):
        super().__init__(layer=layer, **kwargs)
        self._ln_2 = math.log(2)  # constant ln(2)

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # gradient of log2(x) = 1 / (x * ln(2))
        return gradient / (input * self._ln_2)


def get_backward_Log2(layer: Log2) -> Layer:
    """
    This function creates a `BackwardLog2` layer based on a given `Log2` layer.
    It provides a convenient way to obtain the backward pass of the input `Log2` layer,
    using the `BackwardLog2`.

    ### Parameters:
    - `layer`: A Keras `Log2` layer instance.
      The function uses this layer's configurations to set up the `BackwardLog2` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardLog2`, which acts as the reverse layer for the given `Log2`.

    ### Example Usage:
    ```python
    from keras.layers import Log2
    from keras_custom.backward import get_backward_Log2

    # Assume `log2_layer` is a pre-defined Log2 layer
    backward_layer = get_backward_Log2(log2_layer)
    output = backward_layer(input_tensor)
    ```
    """
    return BackwardLog2(layer)
