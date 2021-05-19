from keras.layers import Conv2D, BatchNormalization, Activation, Add


def residual_block(
    x,
    filters: int,
    padding: str,
    normalization: bool,
    layers_before_pooling: int,
    name=None,
):
    """Function implementing an Identity Residual block."""

    x = Conv2D(filters, 3, padding=padding, kernel_initializer="he_normal")(x)

    for i in range(layers_before_pooling):
        x_shortcut = x
        x = Conv2D(filters, 3, padding=padding, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, 3, padding=padding, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = Add()([x, x_shortcut])

        if name is None:
            x = Activation("relu")(x)
        else:
            x = Activation("relu", name="conv_{}_{}".format(name, i))(x)

    return x


def conv_block(
    x,
    filters: int,
    padding: str,
    normalization: bool,
    layers_before_pooling: int,
    name=None,
):

    for i in range(layers_before_pooling):

        x = Conv2D(filters, 3, padding=padding, kernel_initializer="he_normal")(x)

        if normalization:
            x = BatchNormalization()(x)

        if name is None:
            x = Activation("relu")(x)
        else:
            x = Activation("relu", name="conv_{}_{}".format(name, i))(x)
    return x


blocks = {"standard": conv_block, "residual": residual_block}
