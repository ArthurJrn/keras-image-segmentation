import models
from keras import Model


def build_unet(**kwargs) -> Model:
    """Helper function that builds a Unet model instance.

    Args:
        **kwargs (dict): dict of model parameters

    Returns:
        model (keras.Model): Unet built model

    """
    input_size = (
        kwargs["input_size"]["width"],
        kwargs["input_size"]["height"],
        kwargs["input_size"]["channels"],
    )
    unet = models.Unet(
        block_type=kwargs["block_type"],
        padding_type=kwargs["padding_type"],
        pooling_number=kwargs["pooling_number"],
        layers_before_pooling=kwargs["layers_before_pooling"],
        normalization=kwargs["normalization"],
        input_size=input_size,
    )

    model = unet.create_model()
    return model


def get_build_function(name: str):
    """Helper function that returns the building function associated to a model name. Useful if other models are added.

    Args:
        name (str): name of the model to build

    Returns:
        (function): model building function
    """
    if name not in AVAILABLE_MODELS:
        raise NotImplementedError(name)
    else:
        return AVAILABLE_MODELS[name]


# This dict contains all the available models
AVAILABLE_MODELS = {"unet": build_unet}
