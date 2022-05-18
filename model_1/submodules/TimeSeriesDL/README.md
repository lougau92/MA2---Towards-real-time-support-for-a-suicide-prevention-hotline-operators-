# Time-Series prediction: First assignment of Deep Learning
Repository of the first assignment of the deep learning course (University Maastricht).

## Setup
Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) on your system and execute the follwing command afterwards.

```$ conda env create -f environment.yml```

After installation, the environment can be activated by calling 

```$ conda activate dl-1```

## Usage
### Training
This program uses configuration files to set program arguments and deep learning hyperparameters. To configure a file, have a look at the default files located in ```config/```. To start a training, call the following command (the config file path might differ)

```$ python main.py --config config/default_lstm.yml```

Now, the training is running and a log folder is created in the directory ```runs/<MODEL_TYPE>/<TIME_STAMP>```. Every log folder contains the configuration which was used to start a training. By this, it is easier to keep track of the best hyperparameters found so far.

### Tensorboard
Make sure the conda environment is enabled, then call

```$ tensorboard --logdir=runs```

to show all trainings in tensorboard. Press [here](http://localhost:6006) to access the webpage.

## Configuration
The configuration uses the YAML language and is generally built like

```
device: cpu
model_name: REGISTERED_MODEL_NAME
model_args:
  ...
dataset_args:
  d_type: DATASET_NAME
  normalize: True
  bounds: [0, 1]
  sequence_length: 200
  future_steps: 1
dataloader_args:
  num_workers: 1
  batch_size: 4
  shuffle: True
train_epochs: 10
evaluation: None
```

## Features
- [x] GPU support for CUDA capable graphic cards
- [x] Configuration management
- [x] LSTM
- [x] Use convolutional layers before passing values into the LSTM
- [ ] GRU
- [x] Xavier weight initialization
- [x] ReLU, Sigmoid & Tanh
- [ ] Dropout
- [x] Variable learning rate (using decay)
- [x] Tensorboard logging, model saving & more
- [x] Dataset normalization & transformation
- [x] Dataset random shuffle
- [x] Split into train- / validationset
- [x] Prediction of 'x' steps into future
- [x] visualization of dataset / prediction
