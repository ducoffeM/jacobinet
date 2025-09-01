# abstract class for BackwardLayer
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import keras
import keras.ops as K  # type:ignore
from jacobinet.layers.utils import reshape_to_batch, share_weights_and_build
from jacobinet.utils import to_tuple
from keras import KerasTensor as Tensor
from keras.layers import Activation, Input, Layer  # type:ignore


@keras.saving.register_keras_serializable()
class BackwardLayer(Layer):
    """
    A custom Keras wrapper layer that reverses the operations of a given layer.

    `BackwardLayer` is designed to perform the reverse operation of a wrapped Keras
    layer during the backward pass, which can be useful for certain architectures
    requiring layer operations to be undone or applied in reverse.

    Usage:
    ------
    ```python
    from keras.layers import Dense
    backward_layer = BackwardLayer(layer=Dense(32))
    ```

    This layer can be useful in advanced neural network architectures that
    need custom layer reversals, such as autoencoders, invertible networks,
    or specialized backpropagation mechanisms.

    Notes:
    ------
    - The `BackwardLayer` does not automatically compute a true inverse; it only
      reverses the application of operations as defined by the wrapped layer.
    - It requires the wrapped layer to have compatible reverse operations.
    """

    def __init__(
        self,
        layer: Layer,
        input_dim_wo_batch: Union[None, Tuple[int]] = None,
        output_dim_wo_batch: Union[None, Tuple[int]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.layer = layer
        if input_dim_wo_batch is None:
            if isinstance(layer.input, list):
                input_dim_wo_batch = tuple(input_i.shape[1:] for input_i in self.layer.input)
            else:
                input_dim_wo_batch = self.layer.input.shape[1:]

        self.input_dim_wo_batch = input_dim_wo_batch
        if isinstance(self.input_dim_wo_batch[0], tuple):
            self.n_input = len(input_dim_wo_batch)
        else:
            self.n_input = 1
        if output_dim_wo_batch is None:
            output_dim_wo_batch = self.layer.output.shape[1:]
        self.output_dim_wo_batch = output_dim_wo_batch

        # In many scenarios, the backward pass can be represented using a native Keras layer
        # (e.g., Conv2D can map to Conv2DTranspose).
        # In such cases, users can directly specify a `layer_backward` function,
        # which will be invoked automatically.
        self.layer_backward: Optional[Callable[..., Any]] = None

    def get_config(self) -> Dict:
        config = super().get_config()
        layer_config = keras.saving.serialize_keras_object(self.layer)
        # self.constant is a tensor, first convert it to float value
        dico_params = {}
        dico_params["layer"] = layer_config
        # save input shape
        dico_params["input_dim_wo_batch"] = keras.saving.serialize_keras_object(
            self.input_dim_wo_batch
        )
        dico_params["output_dim_wo_batch"] = keras.saving.serialize_keras_object(
            self.output_dim_wo_batch
        )

        config.update(dico_params)

        return config

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Union[Tensor, List[Tensor]]:
        if self.layer_backward:
            return self.layer_backward(gradient)
        raise NotImplementedError()

    def build(self, input_shape: tuple[int]) -> None:
        super().build(input_shape)

    def call(
        self,
        inputs: Union[Tensor, List[Tensor]],
        training: Union[bool, None] = None,
        mask: Union[None, Tensor] = None,
    ) -> Union[Tensor, List[Tensor]]:
        layer_input = None

        if not isinstance(inputs, list):
            gradient = inputs
        else:
            gradient = inputs[0]
            if len(inputs) == 2:
                layer_input = inputs[1]
            elif len(inputs) > 2:
                layer_input = inputs[1:]
        reshape_tag, gradient_, n_out = reshape_to_batch(gradient, (1,) + self.output_dim_wo_batch)
        output = self.call_on_reshaped_gradient(
            gradient_, input=layer_input, training=training, mask=mask
        )

        if reshape_tag:
            if isinstance(output, list):
                output = [
                    K.reshape(
                        output[i],
                        (-1,) + to_tuple(n_out) + to_tuple(self.input_dim_wo_batch[i]),
                    )
                    for i in range(self.n_input)
                ]
            else:
                output = K.reshape(output, (-1,) + tuple(n_out) + self.input_dim_wo_batch)

        return output

    @classmethod
    def from_config(cls: Any, config: Dict) -> Type[Layer]:
        layer_config = config.pop("layer")
        layer = keras.saving.deserialize_keras_object(layer_config)

        input_dim_wo_batch_config = config.pop("input_dim_wo_batch")
        input_dim_wo_batch = keras.saving.deserialize_keras_object(input_dim_wo_batch_config)
        if isinstance(input_dim_wo_batch, list):
            if isinstance(input_dim_wo_batch[0], list):
                input_dim_wo_batch = tuple(
                    tuple(input_dim_wo_batch_i) for input_dim_wo_batch_i in input_dim_wo_batch
                )
            else:
                input_dim_wo_batch = tuple(input_dim_wo_batch)

        output_dim_wo_batch_config = config.pop("output_dim_wo_batch")
        output_dim_wo_batch = keras.saving.deserialize_keras_object(output_dim_wo_batch_config)
        if isinstance(output_dim_wo_batch, list):
            if isinstance(output_dim_wo_batch[0], list):
                output_dim_wo_batch = tuple(
                    tuple(output_dim_wo_batch_i) for output_dim_wo_batch_i in output_dim_wo_batch
                )
            else:
                output_dim_wo_batch = tuple(output_dim_wo_batch)

        return cls(
            layer=layer,
            input_dim_wo_batch=input_dim_wo_batch,
            output_dim_wo_batch=output_dim_wo_batch,
            **config,
        )

    def compute_output_shape(
        self, input_shape: Tuple[Union[int, Tuple[int]]]
    ) -> Union[Tuple[Tuple[int, ...], ...], Tuple[int, ...]]:
        if isinstance(self.input_dim_wo_batch[0], tuple):
            return tuple(
                (1,)
                + (
                    input_dim_wo_batch_i
                    if isinstance(input_dim_wo_batch_i, tuple)
                    else (input_dim_wo_batch_i,)
                )
                for input_dim_wo_batch_i in self.input_dim_wo_batch
            )
        return (None,) + self.input_dim_wo_batch


@keras.saving.register_keras_serializable()
class BackwardLinearLayer(BackwardLayer):
    """
    A custom Keras wrapper layer that reverses the operations of a given layer.

    `BackwardLayer` is designed to perform the reverse operation of a wrapped Keras
    layer during the backward pass, which can be useful for certain architectures
    requiring layer operations to be undone or applied in reverse.

    Usage:
    ------
    ```python
    from keras.layers import Dense
    backward_layer = BackwardLayer(layer=Dense(32))
    ```

    This layer can be useful in advanced neural network architectures that
    need custom layer reversals, such as autoencoders, invertible networks,
    or specialized backpropagation mechanisms.

    Notes:
    ------
    - The `BackwardLayer` does not automatically compute a true inverse; it only
      reverses the application of operations as defined by the wrapped layer.
    - It requires the wrapped layer to have compatible reverse operations.
    """


@keras.saving.register_keras_serializable()
class BackwardNonLinearLayer(BackwardLayer):
    """
    A custom Keras wrapper layer that reverses the operations of a given layer.

    `BackwardNonLinearLayer` is designed to perform the reverse operation of a wrapped Keras
    layer during the backward pass, which can be useful for certain architectures
    requiring layer operations to be undone or applied in reverse.

    Usage:
    ------
    ```python
    from keras.layers import Dense
    backward_layer = BackwardLayer(layer=Dense(32))
    ```

    This layer can be useful in advanced neural network architectures that
    need custom layer reversals, such as autoencoders, invertible networks,
    or specialized backpropagation mechanisms.

    Notes:
    ------
    - The `BackwardLayer` does not automatically compute a true inverse; it only
      reverses the application of operations as defined by the wrapped layer.
    - It requires the wrapped layer to have compatible reverse operations.
    """


@keras.saving.register_keras_serializable()
class BackwardBoundedLinearizedLayer(BackwardLinearLayer):
    """
    A custom Keras wrapper layer to linearize operators with bounded derivatives.

    TO DO

    `BackwardNonLinearLayer` is designed to perform the reverse operation of a wrapped Keras
    layer during the backward pass, which can be useful for certain architectures
    requiring layer operations to be undone or applied in reverse.

    Usage:
    ------
    ```python
    from keras.layers import Dense
    backward_layer = BackwardLayer(layer=Dense(32))
    ```

    This layer can be useful in advanced neural network architectures that
    need custom layer reversals, such as autoencoders, invertible networks,
    or specialized backpropagation mechanisms.

    Notes:
    ------
    - The `BackwardLayer` does not automatically compute a true inverse; it only
      reverses the application of operations as defined by the wrapped layer.
    - It requires the wrapped layer to have compatible reverse operations.
    """


@keras.saving.register_keras_serializable()
class BackwardWithActivation(BackwardNonLinearLayer):
    """
    This class implements a custom layer for backward pass of a layer in Keras with a non linear activation function.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python

    # Assume `layer` is a pre-defined Keras layer with a linear function followed by non linear activation
    backward_layer = BackwardWithActivation(layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Layer,
        backward_linear: Type[BackwardLinearLayer],
        backward_activation: Type[BackwardNonLinearLayer],
        **kwargs: Any,
    ):
        super().__init__(layer=layer, **kwargs)
        activation_name = layer.get_config()["activation"]
        self.activation_backward = backward_activation(
            Activation(activation_name),
            input_dim_wo_batch=self.output_dim_wo_batch,
            output_dim_wo_batch=self.output_dim_wo_batch,
        )

        dico_config = self.layer.get_config()
        dico_config.pop("activation")
        cls = self.layer.__class__
        self.layer_wo_activation = cls.from_config(dico_config)
        layer_wo_activation_tmp = cls.from_config(
            dico_config
        )  # create a temporary layer w/o activation
        layer_wo_activation_tmp(Input(self.input_dim_wo_batch))
        weights_names = [w.name for w in layer_wo_activation_tmp.weights]
        share_weights_and_build(
            original_layer=self.layer,
            new_layer=self.layer_wo_activation,
            weight_names=weights_names,
            input_shape_wo_batch=self.input_dim_wo_batch,
        )

        self.layer_backward: Callable[..., Any] = backward_linear(
            self.layer_wo_activation,
            input_dim_wo_batch=self.input_dim_wo_batch,
            output_dim_wo_batch=self.output_dim_wo_batch,
        )

        self.layer_wo_activation.built = True

    def call(
        self,
        inputs: Union[Tensor, List[Tensor]],
        training: Union[bool, None] = None,
        mask: Union[None, Tensor] = None,
    ) -> Union[Tensor, List[Tensor]]:
        # apply locally the chain rule
        # (f(g(x)))' = f'(x)*g'(f(x))
        # compute f(x) as inner_input

        gradient = inputs[0]
        input = inputs[1]
        inner_input = self.layer_wo_activation(input)
        # computer gradient*g'(f(x))
        backward_output: Tensor = self.activation_backward(inputs=[gradient, inner_input])
        # compute gradient*g'(f(x))*f'(x)
        output = self.layer_backward(inputs=[backward_output])

        return output
