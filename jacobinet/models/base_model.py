from typing import Any

import keras
from jacobinet.layers.layer import BackwardLayer

from .utils import is_linear_layer


@keras.saving.register_keras_serializable()
class BackwardModel(keras.Model):
    def __init__(
        self,
        *args,
        **kwargs: Any,
    ):
        """
        A custom Keras model that supports backward computation for layers and models.

        This class extends the Keras `Model` class and is designed to perform backward computations
        for a given set of layers or sub-models. It includes an `n_input` attribute, which determines
        the number of inputs to the model, and checks whether the model consists of linear layers
        based on its layers.

        Example:
            ```python
            model = BackwardModel(n_input=3, layers=[...])
            print(model.n_input)  # Prints 3
            ```
        """
        if "n_input" in kwargs:
            self.n_input = kwargs.pop("n_input")
        else:
            self.n_input = 1
        super().__init__(*args, **kwargs)
        self.is_linear = min(
            [
                is_linear_layer(layer)
                for layer in self.layers
                if isinstance(layer, BackwardLayer)
                or isinstance(layer, BackwardModel)
                or isinstance(layer, BackwardSequential)
            ]
        )

    def get_config(self):
        config = super().get_config()
        config["n_input"] = self.n_input

        return config


@keras.saving.register_keras_serializable()
class BackwardSequential(keras.Sequential):
    """
    A custom Keras Sequential model that supports backward computations and checks for linear layers.

    This class extends Keras' `Sequential` model to support backward computations for layers in the model.
    It tracks whether the model contains any linear layers through the `is_linear` attribute and allows adding
    and removing layers while automatically updating this attribute. The model also supports configuring the
    number of inputs (`n_input`), which is set to 1 by default.

    Example:
        ```python
        model = BackwardSequential(layers=[...])
        model.add(new_layer)
        model.pop()
        ```
    """

    n_input = 1
    is_linear = True

    def add(self, layer, rebuild=True):
        """Adds a layer instance on top of the layer stack.

        Args:
            layer: layer instance.
        """
        super().add(layer=layer, rebuild=rebuild)
        # recompute is_linear
        self.is_linear = min([self.is_linear, is_linear_layer(layer)])

    def pop(self, rebuild=True):
        """Removes the last layer in the model."""
        super().pop(rebuild=rebuild)
        self.is_linear = min([is_linear_layer(layer) for layer in self.layers])

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        backward_model = super().from_config(config)
        return backward_model
