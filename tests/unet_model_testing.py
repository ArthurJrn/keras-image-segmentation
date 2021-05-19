import unittest
import models
from build_model import build_unet
import numpy as np


class TestUnetModel(unittest.TestCase):
    def test_image_output_dimension(self):
        unet = models.Unet(
            block_type="residual",
            padding_type="same",
            pooling_number=3,
            layers_before_pooling=2,
            normalization=True,
            start_filters=64,
            input_size=(256, 256, 1),
        )
        model = unet.create_model()
        x = np.zeros((256, 256, 1), np.uint8)
        x = np.expand_dims(x, axis=0)
        output = model.predict(x)
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], 256)
        self.assertEqual(output.shape[2], 256)

    def test_build_model(self):
        unet = models.Unet(
            block_type="residual",
            padding_type="same",
            pooling_number=3,
            layers_before_pooling=2,
            normalization=True,
            start_filters=64,
            input_size=(256, 256, 1),
        )
        model = unet.create_model()
        built_model = build_unet(
            **{
                "block_type": "residual",
                "padding_type": "same",
                "pooling_number": 3,
                "layers_before_pooling": 2,
                "normalization": True,
                "start_filters": 64,
                "input_size": {"width": 256, "height": 256, "channels": 1},
            }
        )
        for l1, l2 in zip(model.layers, built_model.layers):
            model_config = l1.get_config()
            built_model_config = l2.get_config()

            #   Remove the name as it is not relevant to compare
            del model_config["name"]
            del built_model_config["name"]
            assert model_config == built_model_config


if __name__ == "__main__":
    unittest.main()
