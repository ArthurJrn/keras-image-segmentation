# Script to visualize Data Augmentation
from data import create_train_generator
import yaml
import numpy as np
import matplotlib.pyplot as plt


def visualize_data_augmentation(config_path) -> None:
    """Quick function that allow to visualize by hands the data augmentation defined in the config file

    Args:
        config_path: str
            path to the config file

    Returns:
        None
    """
    # Load Yaml config file
    with open(config_path) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    # Get data augmentation
    augmentations = parameters["DATA_AUGMENTATION_PARAMS"]
    nb_augmentation = len(augmentations)

    # Dataset path
    path = parameters["DATASET"]["TRAIN_PATH"]

    # Create train generator
    myGene = create_train_generator(1, path, "image", "label", augmentations)

    # Visualize
    fig, ax = plt.subplots(nrows=2, ncols=nb_augmentation, figsize=(15, 15))

    for i in range(nb_augmentation):
        (img, mask) = next(myGene)
        # convert to unsigned integers
        image = img[0]
        mask = mask[0]
        ax[0, i].imshow(image)
        ax[1, i].imshow(mask)
        ax[0, i].axis("off")
        ax[1, i].axis("off")

    plt.savefig("image.png")


# Parameters for predictions visualization
Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array(
    [
        Sky,
        Building,
        Pole,
        Road,
        Pavement,
        Tree,
        SignSymbol,
        Fence,
        Car,
        Pedestrian,
        Bicyclist,
        Unlabelled,
    ]
)
