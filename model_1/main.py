import argparse
from multiprocessing import freeze_support
import torch
import math
import copy
import numpy as np
import matplotlib.pyplot as plt


from submodules.TimeSeriesDL.utils.config import config

#from model.brain_behaviour_classifier import BrainBehaviourClassifier

from torch import nn
from torch.autograd import Variable


from torch.utils.data import DataLoader
from data.dataset import CDS_Dataset

from tqdm import tqdm


#config.register_model("BrainBehaviourClassifier", BrainBehaviourClassifier)
#config.register_model("LinearModel", Linear)
config_dict = None


class RegressionModel(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(RegressionModel, self).__init__()
        self.dense_h1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.relu_h1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.dense_out = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, X):

        out = self.relu_h1(self.dense_h1(X))
        out = self.dropout(out)
        out = self.dense_out(out)

        return out



def train():
    # define parameters (depending on device, lower the precision to save memory)
    device = config_dict["device"]
    precision = torch.float16 if device == "cuda" else torch.float32

    # load the data, normalize them and convert them to tensor
    dataset = CDS_Dataset(**config_dict["dataset_args"])

    split_sizes = [int(math.ceil(len(dataset) * 0.8)), int(math.floor(len(dataset) * 0.2))]
    

    trainset, valset = torch.utils.data.random_split(dataset, split_sizes)
    #trainloader = DataLoader(trainset, **config_dict["dataloader_args"])
    #valloader = DataLoader(valset, **config_dict["dataloader_args"])
    

    input_size = 266
    hidden_layer_size = 1
    learning_rate = 0.05
    batch_size = 50
    num_epochs = 100

    m = RegressionModel(input_size=input_size, hidden_size=hidden_layer_size)
    cost_func = nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)


    all_losses = []
    for e in tqdm(range(num_epochs)):
        batch_losses = []

        for ix, (Xb, yb) in enumerate(trainset):
            _X = Variable(Xb).float()
            _y = Variable(yb).float()

            #==========Forward pass===============

            preds = m(_X)
            loss = cost_func(preds, _y)

            #==========backward pass==============

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.data)
            all_losses.append(loss.data)

        mbl = np.mean(np.sqrt(batch_losses)).round(3)

        if e % 5 == 0:
            print("Epoch [{}/{}], Batch loss: {}".format(e, num_epochs, mbl))


    print(m.training)
    m.eval()
    print(m.training)

if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="This program trains and tests a deep " +
                                                 "learning model to regress on CDS data")
    parser.add_argument("--config", dest="config", help="Set path to config file.")

    args = parser.parse_args()


    if args.config:
        config_dict = config.get_args(args.config)
        train()
    else:
        raise ValueError("Config file not set. Use '--config <path_to_file>' to load a configuration.")
