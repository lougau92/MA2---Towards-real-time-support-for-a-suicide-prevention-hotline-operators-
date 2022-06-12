import argparse
from multiprocessing import freeze_support
import torch
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from submodules.TimeSeriesDL.utils.config import config

from torch import nn
from torch.autograd import Variable

from torch.utils.data import DataLoader
from data.dataset import CDS_Dataset

from model.linear_regression import MLP

#config.register_model("LinearModel", Linear)
config_dict = None


def train():
    # define parameters (depending on device, lower the precision to save memory)
    device = config_dict["device"]
    precision = torch.float16 if device == "cuda" else torch.float32

    # load the data, normalize them and convert them to tensor
    dataset = CDS_Dataset(**config_dict["dataset_args"])

    split_sizes = [int(math.ceil(len(dataset) * 0.8)), int(math.floor(len(dataset) * 0.2))]
    trainset, valset = torch.utils.data.random_split(dataset, split_sizes)

    trainloader = DataLoader(trainset, **config_dict["dataloader_args"])
    valloader = DataLoader(valset, **config_dict["dataloader_args"])
    

    print(f"Dataset length: {len(dataset)}")
    print(f"Trainloader length: {len(trainloader)}")
    print(f"Valloader length: {len(valloader)}")


    model = MLP(lr=1e-3, lr_decay=9e-1)
    model.use_device(device)

    # model.train(trainloader, epochs=5, two_loss_functions=True)
    # model.validate(valloader)
    model.learn(loader=trainloader, validate=valloader, test=None, epochs=config_dict["train_epochs"])


def get_pre_eval(id, negative=True):
    df = pd.read_csv(os.path.join("../filtered_conv", "prechat_questions.tsv"), sep='\t')
    df = df[df["event_id"] == id]
    df = df.drop(columns=["event_id"])

    if negative:
        # We drop "I have a will to live column and average the other numbers"
        df = df.drop(columns=["Ik heb de wil om te leven"])
        y = df.mean(axis = 1)
        y = y.values.astype(np.float32)[0]
    else:
        y = df['Ik heb de wil om te leven']
        y = y.values.astype(np.float32)[0]

    return y

def get_post_eval(id, negative=True):
    df = pd.read_csv(os.path.join("../filtered_conv", "postchat_questions.tsv"), sep='\t')
    df = df[df["event_id"] == id]
    df = df.drop(columns=["event_id"])

    if negative:
        # We drop "I have a will to live column and average the other numbers"
        df = df.drop(columns=["Ik heb de wil om te leven"])
        y = df.mean(axis = 1)
        y = y.values.astype(np.float32)[0]
    else:
        y = df['Ik heb de wil om te leven']
        y = y.values.astype(np.float32)[0]

    return y



def test():
    device = config_dict["device"]
    precision = torch.float16 if device == "cuda" else torch.float32

    # We load positive and negative models
    
    model_pos = MLP(lr=1e-3, lr_decay=9e-1)
    model_pos.use_device(device)
    model_pos.load_state_dict(torch.load("./runs/Linear_Regression/p2/p2.torch"))
    model_pos.eval()

    model_neg = MLP(lr=1e-3, lr_decay=9e-1)
    model_neg.use_device(device)
    model_neg.load_state_dict(torch.load("./runs/Linear_Regression/n1/n1.torch"))
    model_neg.eval()    
   

    convos = os.listdir("../filtered_conv")
    f = "cl000060.csv"
    # 44, 56, 68
    df = pd.read_csv(os.path.join("../filtered_conv", f), sep='\t')
    df = df.drop(columns=["event_id","message_id", "Unnamed: 0", "Unnamed: 0.1", "user_handle"])
    df = df.astype(int)

    pre_neg = get_pre_eval(f[:-4], negative=True)
    pre_pos = get_pre_eval(f[:-4], negative=False)

    post_neg = get_post_eval(f[:-4], negative=True)
    post_pos = get_post_eval(f[:-4], negative=False)
    print(f"Pre neg eval: {pre_neg}")
    print(f"Post neg eval: {post_neg}")

    print(f"Pre pos eval: {pre_pos}")
    print(f"Post pos eval: {post_pos}")

    # Rolling window through the conversation
    num = int(0.3 * len(df.index))
    pos_values = list()
    neg_values = list()
    conv_time = list()
    conv_length =  df['sec_since_start'].iloc[-1]

    for i in range(0,len(df.index) - num):
        d = df.iloc[i:i+num]
        d = d.mean(axis = 0)

        x = d.values.astype(np.float32)
        x[-1] = x[-1] / conv_length
        xt = torch.from_numpy(x).float()

        with torch.no_grad():
            xt = xt.to(device)

            eval_pos = model_pos(xt)
            eval_neg = model_neg(xt)

            pos_values.append(eval_pos.cpu().detach().numpy()[0])
            neg_values.append(eval_neg.cpu().detach().numpy()[0])
        
        conv_time.append((i+num)/len(df.index))


        
    print(neg_values)
    print(pos_values)


    plt.plot(conv_time,pos_values, label = "Positive state of mind")
    plt.plot(conv_time,neg_values, label = "Negative state of mind")
    plt.plot(conv_time[0],[pre_neg], marker="o", markersize=20,  markerfacecolor="orange")
    plt.plot(conv_time[-1],[post_neg], marker="o", markersize=20,  markerfacecolor="orange")
    plt.plot(conv_time[0],[pre_pos], marker="o", markersize=20,  markerfacecolor="blue")
    plt.plot(conv_time[-1],[post_pos], marker="o", markersize=20,  markerfacecolor="blue")
    plt.legend()
    plt.title("Number of messages in window: " + str(num))
    plt.show()


    return 0


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="This program trains and tests a deep " +
                                                 "learning model to regress on CDS data")
    parser.add_argument("--config", dest="config", help="Set path to config file.")
    args = parser.parse_args()

    if args.config:
        config_dict = config.get_args(args.config)
        # test()
        # train()
    #else:
        #raise ValueError("Config file not set. Use '--config <path_to_file>' to load a configuration.")