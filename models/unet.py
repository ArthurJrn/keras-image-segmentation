from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, Input, concatenate
from keras import Model
from .model_utils import blocks


class Unet:
    """
    A class used to build an Unet model.

    ...

    Attributes
    ----------
    block_type : str
        a string defining which block to use in the model. Available blocks can be found in model_utils
    padding_type : str in ["valid", "same"]
        a string defining the type of padding in the convolutions
    pooling_number : int
        an int defning the number of pooling operations, ie the number of blocks
    layers_before_pooling : int
        the number of layers in a block
    normalization: bool
        bool defining whether or not adding a batch normalization layer in the block.
    start_filters: int
        int defining the size of the first convolutionnal layer"s filter
    intput_size: (int, int, int)
        shape of the input of the model

    Methods
    -------
    create_model
        Return a Unet model with the given parameters
    """

    def __init__(
        self,
        block_type: str,
        padding_type: str,
        pooling_number: int,
        layers_before_pooling: int,
        normalization: bool,
        start_filters=64,
        input_size=(256, 256, 1),
    ):
        self.block_type = block_type
        self.padding_type = padding_type
        self.pooling_number = pooling_number
        self.layers_before_pooling = layers_before_pooling
        self.normalization = normalization
        self.input_size = input_size
        self.start_filters = start_filters

    def create_model(self):
        """Returns a Unet model with the parameters of the class.

        Returns:
            model (keras.Model): Keras Model of the created Unet. Model is uncompiled.
        """

        # Get block builder function
        block = blocks[self.block_type]

        # Start of the model building
        inputs = Input(self.input_size)
        x = inputs

        # We build the model iteratively: each block if followed by a pooling layer
        # After each pooling layer, filters are multiplied by 2
        for i in range(self.pooling_number):
            x = block(
                x,
                self.start_filters * (2 ** i),
                self.padding_type,
                self.normalization,
                self.layers_before_pooling,
                i,
            )
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # The encoder part of the Unet is build
        # Middle of the model: convolution layers with dropout
        x = Conv2D(
            self.start_filters * (2 ** self.pooling_number),
            3,
            activation="relu",
            padding=self.padding_type,
            kernel_initializer="he_normal",
        )(x)
        x = Conv2D(
            self.start_filters * (2 ** self.pooling_number),
            3,
            activation="relu",
            padding=self.padding_type,
            kernel_initializer="he_normal",
        )(x)
        drop = Dropout(0.5)(x)

        # We build the encoder model to be able to acces layers by their name, necessary the concatenation layers
        encoder = Model(inputs, drop)

        x = drop

        # Second part of the Unet architecture, symmetrical to the first part
        # After each upsampling operation, filters are divided by 2
        for j in range(self.pooling_number):
            x = Conv2D(
                self.start_filters * 2 ** (self.pooling_number - j - 1),
                2,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            )(UpSampling2D(size=(2, 2))(x))
            layer_to_merge = encoder.get_layer(
                "conv_{}_{}".format(
                    self.pooling_number - j - 1, self.layers_before_pooling - 1
                )
            ).output
            x = concatenate([layer_to_merge, x], axis=3)
            x = block(
                x,
                self.start_filters * 2 ** (self.pooling_number - j - 1),
                "same",
                False,
                self.layers_before_pooling,
            )

        # Ouput layers
        x = Conv2D(
            2,
            3,
            activation="relu",
            padding=self.padding_type,
            kernel_initializer="he_normal",
        )(x)
        output = Conv2D(1, 1, activation="sigmoid")(x)

        # Final creation of the model
        model = Model(inputs, output)

        return model
