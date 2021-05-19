from models import blocks
from build_model import AVAILABLE_MODELS
import yaml
import os
import keras.metrics, keras.losses


def check_dataset_architecture(dataset_path: str, dataset_type: str) -> None:
    """Function that checks the path and architecture of a dataset.
    Args:

        dataset_path (str): path to the dataset
        dataset type (str): type of dataset, train or test.

    Raises:
        FileNotFoundError: if the dataset path or architecture is wrong.

    Returns:
        None
    """
    # If a dataset path is "", it means it is not used during the training or testing so we don"t raise an error
    if not dataset_path:
        pass
    else:
        if dataset_type == "train":
            assert (
                os.path.exists(dataset_path)
                and os.path.exists(dataset_path + "/images")
                and os.path.exists(dataset_path + "/masks")
            ), "The dataset folder {} does not exist or have wrong tree architecture".format(
                dataset_path
            )
        else:
            assert os.path.exists(
                dataset_path
            ), "The dataset folder {} does not exist".format(dataset_path)


def test_config_dict(config: dict) -> None:
    """Function that checks the values of the yaml config file used for main.py.

    Args:
        config (dict): dict of the loaded yaml config file.
        dataset type (str): type of dataset, train or test.

    Raises:
        FileNotFoundError: if a specified path does not exists or if a dataset has wrong architecture.

        NotImplementedError: if a model name or a block type is not implemented

    Returns: None
    """

    # Checks that the config file has the right keywords
    model_name = config["model"]["name"]
    assert model_name in AVAILABLE_MODELS, "The model {} is not implemented".format(
        model_name
    )

    # Check model params
    block_type = config["model"]["block_type"]
    assert block_type in blocks, "The block type {} is not implemented".format(
        block_type
    )

    # Check dataset paths
    # Our datasets need to have this tree architecture:
    # -dataset:
    #   - image
    #   - label
    train_path = config["dataset"]["train_path"]
    test_path = config["dataset"]["test_path"]

    # Check the dataset paths and architecture
    check_dataset_architecture(train_path, "train")
    check_dataset_architecture(test_path, "test")

    # Chec results save path
    result_save_path = config["result_path"]
    assert result_save_path and os.path.exists(
        result_save_path
    ), "The results folder {} doest not exist".format(result_save_path)

    # Check log dir
    log_dir_path = config["log_dir"]
    assert result_save_path and os.path.exists(
        log_dir_path
    ), "The logs folder {} does not exist.".format(log_dir_path)
