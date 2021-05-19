from data import create_train_generator, create_test_generator, save_result
from build_model import get_build_function
from train import train_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import argparse
import yaml
from datetime import datetime
from config_checking import test_config_dict
import sys
import os
import logging
import numpy as np
from visualize import COLOR_DICT

if __name__ == "__main__":

    # Set logging
    # logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to the config file")
    args = parser.parse_args()

    # Load Yaml config file
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Test the config file
    test_config_dict(config)

    # Load model parameters
    model_params = config["model"]

    # Load training hyperparameters
    hyperparams = config["hyperparameters"]
    l_r = hyperparams["lr"]
    batch_size = hyperparams["batch_size"]
    loss = hyperparams["loss"]
    metrics = hyperparams["metrics"]
    epochs = hyperparams["epochs"]
    steps_per_epoch = hyperparams["steps_per_epoch"]

    # Build model
    model_name = model_params["name"]
    building_function = get_build_function(model_name)
    model = building_function(**model_params)

    if config["mode"] == "train":

        # Get train dataset path
        train_dataset_path = config["dataset"]["train_path"]

        # Compile the model
        model.compile(optimizer=Adam(lr=l_r), loss=loss, metrics=[metrics])

        # Load existing weights if using pretrained weights
        if config["pretrained"]:
            weights_path = config["pretrained_weights_path"]
            model.load_weights(pretrained_weights)

        # Test if there is data augmentation
        if config["data_augmentation"]:
            data_aug_params = config["data_augmentation_params"]

            # Create a Keras ImageDataGenerator for train dataet
            train_data_generator = create_train_generator(
                batch_size, train_dataset_path, data_aug_params, save_to_dir=None
            )

            # Start model training
            save_dir = config["result_path"]
            log_dir = config["log_dir"]
            train_model(
                model_name,
                model,
                train_data_generator,
                steps_per_epoch,
                epochs,
                save_dir,
                log_dir,
            )

    else:

        # Inference mode

        # Load the model
        model_path = config["model_path"]
        model = load_model(model_path)

        # Create the ImageDataGenerator instance for the test images we want to predict
        test_path = config["dataset"]["test_path"]
        num_images = len([f for f in os.listdir(test_path)])
        test_generator = create_test_generator(test_path, num_images)

        # Predict and save the prediction maps
        results = model.predict(test_generator)
        result_path = config["result_path"]
        np.save(file=result_path + "predictions_maps", arr=results)

        # If wanted, we can save the output images
        if config["save_predictions_images"]:
            save_result(COLOR_DICT, result_path, results)
