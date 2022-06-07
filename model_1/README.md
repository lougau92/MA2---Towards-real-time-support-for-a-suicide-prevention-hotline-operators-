# Brain behaviour classification: Third assignment of Deep Learning
Repository of the first assignment of the deep learning course (University Maastricht).

## Dataset
Download the brain dataset here: https://surfdrive.surf.nl/files/index.php/s/2MbsgxYwGQjzQys and use the provided password. Extract the zip-file to the data/ folder.

## Setup
Clone repository and initialize submodules.

```$ git clone git@github.com:Huntler/BrainDL.git```

```$ git submodule init```

```$ git submodule update```

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) on your system and execute the follwing command afterwards.

```$ conda env create -f submodules/TimeSeriesDL/requirements/environment.yml```

After installation, the environment can be activated by calling 

```$ conda activate dl-1```

This time, we also need to use h5py, to install this library call

```$ conda install h5py```

## Usage
### Training
This program uses configuration files to set program arguments and deep learning hyperparameters. To configure a file, have a look at the default files located in ```config/```. To start a training, call the following command (the config file path might differ)

```$ python main.py --config config/default_lstm.yml```

Now, the training is running and a log folder is created in the directory ```runs/<MODEL_TYPE>/<TIME_STAMP>```. Every log folder contains the configuration which was used to start a training. By this, it is easier to keep track of the best hyperparameters found so far.

### Tensorboard
Make sure the conda environment is enabled, then call

```$ tensorboard --logdir=runs```

to show all trainings in tensorboard. Press [here](http://localhost:6006) to access the webpage.
