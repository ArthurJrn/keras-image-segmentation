from keras.callbacks import ModelCheckpoint, TensorBoard
from datetime import datetime
import os


def train_model(
    model_name: str,
    model,
    train_data_generator,
    steps_per_epoch: int,
    epochs: int,
    save_dir: str,
    log_dir: str,
) -> None:
    """Function that launches training of a Keras Model.
    Arguments
    ------
    model_name: str
        name of the model to train, used for model saving and logs
    model: Keras Model
        compiled Keras Model instance to be trained
    train_data_generator: keras.preprocessing.image.ImageDataGenerator instance
        ImageDataGenerator instance of the train dataset.
    steps_per_epochs: int
        steps per epochs
    epoches: int
        the number of epochs of the training
    save_dir: str
        path to the directory where the model checkpoints are saved
    log_dir: str:
        path to the directory where the logs are saved, used for Tensorboard

    Returns
    ------
    None
    """
    # Save the model after each epoch
    model_save_name = os.path.join(
        save_dir, model_name + "-model-membrane-{epoch:02d}.hdf5"
    )
    model_checkpoint = ModelCheckpoint(
        model_save_name, monitor="loss", verbose=1, save_best_only=False
    )

    # Add a callback for TensorBoard monitoring
    logdir = os.path.join(
        log_dir, model_name + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    tensorboard_callback = TensorBoard(log_dir=logdir)

    # Training of the model
    model.fit(
        train_data_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[tensorboard_callback, model_checkpoint],
    )
