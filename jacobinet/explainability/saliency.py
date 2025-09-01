from typing import Union

from jacobinet.models import get_backward_model_with_loss
from keras.layers import Layer
from keras.models import Model


def get_saliency_model(
    model: Model,
    loss: Union[str, Layer] = "logits",
    mapping_keras2backward_classes={},
    mapping_keras2backward_losses={},
    **kwargs,
):
    saliency_model = get_backward_model_with_loss(
        model=model,
        loss=loss,
        mapping_keras2backward_classes=mapping_keras2backward_classes,
        mapping_keras2backward_losses=mapping_keras2backward_losses,
        **kwargs,
    )

    return saliency_model
