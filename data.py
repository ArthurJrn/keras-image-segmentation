from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
from skimage import img_as_ubyte
import skimage.transform as trans
import matplotlib.pyplot as plt


def train_preprocessing(img, mask):
    """Small preprocessing on images and masks before training.

    Args:
        img: array
        mask: array

    Returns:
        tuple of array
    """
    if np.max(img) > 1:
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)


def create_train_generator(
    batch_size: int,
    train_dataset_path: str,
    aug_dict: dict,
    image_folder="images",
    mask_folder="masks",
    image_color_mode="grayscale",
    mask_color_mode="grayscale",
    image_save_prefix="augmented_image",
    mask_save_prefix="augmented_mask",
    save_to_dir=None,
    target_size=(256, 256),
    seed=1,
):
    """Returns an iterator for model predictions.

    Arg:
        batch_size: int
            batch size for training
        train_dataset_path: str
            path of the training dataset
        image_folder: str
            name of the train images folder
        masks_folder: str
            name of the mask images folder
        image_color_mode: one of "grayscale", "rgb", "rgba".
            Whether the images will be converted to have 1 or 3 color channels.
        mask_color_mode: one of "grayscale", "rgb", "rgba".
            Whether the masks will be converted to have 1 or 3 color channels.
        save_to_dir: None or str (default: None).
            Allows you to optionally specify a directory to which to save the augmented pictures being generated (useful for visualizing what you are doing).
        image_save_prefix: str
            Prefix to use for filenames of saved images (only relevant if save_to_dir is set).
        masks_save_prefix: str
            Prefix to use for filenames of saved masks (only relevant if save_to_dir is set)
        target_size: tuple of integers
            The dimensions to which all images found will be resized.
        seed: int
            Random seed for data augmentation

    Returns:
        iterator of images and masks
    """

    # Initialize the ImageDataGenerator instances for images and masks
    image_data_generator = ImageDataGenerator(**aug_dict)
    mask_data_generator = ImageDataGenerator(**aug_dict)

    # Fill them with images from the dataset
    image_data_generator = image_data_generator.flow_from_directory(
        train_dataset_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
    )

    mask_data_generator = mask_data_generator.flow_from_directory(
        train_dataset_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed,
    )

    # Combine them for image segmentation training
    train_generator = zip(image_data_generator, mask_data_generator)

    # Preprocessing
    for (img, mask) in train_generator:
        img, mask = train_preprocessing(img, mask)
        yield (img, mask)


def create_test_generator(
    test_path: str, num_image: int, target_size=(256, 256), as_gray=True
):
    """Returns an iterator for model predictions.

    Args:
        test_dataset_path: str
            path to the test dataset
        num_images: int
            number of images in the dataset
        target_size: tuple of integers
            The dimensions to which all images found will be resized
        as_gray: bool
            steps per epochs

    Returns:
        iterator of images
    """
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%d.png" % i), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,))
        img = np.reshape(img, (1,) + img.shape)
        yield img


def label_visualize(num_class, color_dict, img):
    """Helper function for visualization of prediction in the case of multi class."""
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


def save_result(color_dict, save_path, results, flag_multi_class=False, num_class=2):
    """Function to save predictions images.

    Args:
        color_dict: dict
            dict of color parameters
        save_path: str
            path to save the images
        results: numpy array
            array of the predictions maps
        flag_multi_class: bool
            wheter the prediction is multi class, kept for legacy as this repo is not focused on multi class
        num_classes: int
            number of classes
    """
    for i, item in enumerate(results):
        img = (
            labelVisualize(num_class, color_dict, item)
            if flag_multi_class
            else item[:, :, 0]
        )
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img_as_ubyte(img))
