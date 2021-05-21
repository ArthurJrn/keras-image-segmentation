# Implementation of modulable Unet using Keras

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

This repo contains a modulable implémentation of a Unet architecture. You can choose the configuration of the model using a Yaml File. My work in based on [this repo](https://github.com/zhixuhao/unet).

## Overview

### Data

The original dataset is from [isbi challenge](http://brainiac2.mit.edu/isbi_challenge/), and I"ve downloaded it and done the pre-processing.

You can find it in folder data/membrane_dataset.

It is possible to add other datasets, as long as they have the same architecture as the membrane dataset, ie 
```
train/
    ├── images/
    ├── masks/
test/
    ├── images
```

### Data augmentation

The data for training contains 30 512*512 images, which are far not enough to feed a deep learning neural network. I use a module called ImageDataGenerator in keras.preprocessing.image to do data augmentation.


### Model

![img/u-net-architecture.png](img/u-net-architecture.png)

This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.


---

## How to use

### Dependencies

You can install all the librairies and dependencies using the `requirement.txt` file, running the command: 

`pip install -r requirement.txt`

### Run 

The entry point of this repo is the file `main.py`, which requires to be launched with a YAML config file. This yaml config file defines all the modifiable parameters for the training of inference of a model. 

You can run this script locally or in a docker container, which contains all the dependencies, Cuda and CuDNN for training speedup.

List of the most important config parameters you can change:
- mode: "train" or "predict"
- model: 
    - name: name of the model, for now there is oonly unet but it is possible to add others models in the `models` folder, you then need to update the `build_function.py` file accordingly.
    - the others model parameters define the architecture of the model
- data_augmentation: bool, wheter or not to use data augmentation; you can then change, add or delete the data augmentation parameters below. A quick visualization script is provided to visualize the data augmentations on an image. The ImageDataGenerator code in `data.py` can also save augmented images if needed, with modification to the `save_dir` argument.
- result_path: path to save the predictions map and the saved models
- log_dir: path for the logs, useful for Tensorboard monitoring.


#### Local Run

To run locally, use the command:

`python3 main.py config.yaml`

#### Docker run

To build the docker, use the command:

`make build`

To run the docker and the `main.py`, use the command:

`make run`

### Tensorboard monitoring

After launching a training, you can monitor it using TensorBoard with the command:

`tensorboard --logdir <LOG_DIR>`

### Results

During training, the model is saved after each epoch in the specified results folder. During inference, the output prediction maps have the same format as the input and are saved in the file  `predictions_maps.npy` in the same results folder

### Tests

The tests are incomplete (TODO). To run the test for unet model, use the command:

`python3 -m tests.unet_model_testing`

### Ideas for ameliorations

The test module is far from being complete, so this would be the number 1 priority for ameliorations. I would also like to be able to run the training using cloud services (AzureML/AWS/GCC) as the training of deep learning models locally can be very long and raise memory issues. 

## About Keras

Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
supports both convolutional networks and recurrent networks, as well as combinations of the two.
supports arbitrary connectivity schemes (including multi-input and multi-output training).
runs seamlessly on CPU and GPU.
Read the documentation [Keras.io](http://keras.io/)

Keras is compatible with: Python 2.7-3.5.
