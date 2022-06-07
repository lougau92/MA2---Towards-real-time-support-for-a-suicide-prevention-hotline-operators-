from pkgutil import get_data
from typing import List, Tuple
from xmlrpc.client import Boolean
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import scipy.io
import torch
import numpy as np
import os
import pandas as pd


#from .utils import get_dataset_matrix, get_file_label, get_meshes


class CDS_Dataset(torch.utils.data.Dataset):
    def __init__(self, d_type: str = "train", sequence_ratio: int = 10,
                     dataset_path: str = "./data",
                     negative: Boolean=True):
        super(CDS_Dataset, self).__init__()

        self._seq_ratio = sequence_ratio
        self.dataset_path  = dataset_path
        
        # When looking at pre and post evals - look at negative state or positive state
        self.negative = negative

        # We open prechat and postchat questions files and extract out the 
        # conversation id's that have a rating

        self.all_files = os.listdir(self.dataset_path)


        self.pre_ids = list()
        self.post_ids = list()
        self.get_ids()


        self.pre_ids, self.post_ids = self.filter_ids()


        self.pre_eval = pd.read_csv(os.path.join(self.dataset_path, "prechat_questions.tsv"), sep='\t')
        self.post_eval = pd.read_csv(os.path.join(self.dataset_path, "postchat_questions.tsv"), sep='\t')

        self.X = []
        self.y = []

        #self.prepare_X()
    
    def filter_ids(self):
        # Check for each id in pre_ids and post_ids if the file actually exists in dataset_path
        files = os.listdir(self.dataset_path)
        fil = list()
        for f in files:
            f = f[:-4]
            fil.append(f)
        

        pre = list()
        post = list()
        for p in self.pre_ids:
            if p in fil:
                pre.append(p)

        for p in self.post_ids:
            if p in fil:
                post.append(p)   

        return pre, post

    def prepare_X(self):
        print(f"Number of prechat ids: {len(self.pre_ids)}")
        for pre_f in self.pre_ids:
            df = pd.read_csv(os.path.join(self.dataset_path, pre_f + ".csv"), sep='\t')
            l = len(df.index)
            num_msg = int(l * self._seq_ratio)

            df = df.head(num_msg) # WE TAKE THE BEGINNING
            df = df.drop(columns=["event_id","message_id", "Unnamed: 0", "Unnamed: 0.1", "user_handle"])

            df = df.mean(axis = 0)

            x = df.values.astype(np.float32)
            xt = torch.tensor(x).float()
            self.X.append(xt)

        print(f"Number of post ids: {len(self.post_ids)}")
        for post_f in self.post_ids:
            # Open file, get first _seq_ration number of messages,
            # calculate averages for X and Y
            df = pd.read_csv(os.path.join(self.dataset_path, post_f + ".csv"), sep='\t')
            l = len(df.index)
            num_msg = int(l * self._seq_ratio)

            df = df.tail(num_msg) # WE TAKE THE TAIL END
            df = df.drop(columns=["event_id","message_id", "Unnamed: 0", "Unnamed: 0.1", "user_handle"])

            df = df.mean(axis = 0)

            x = df.values.astype(np.float32)
            xt = torch.tensor(x).float()
            self.X.append(xt)
    
    def get_pre_X(self, f):
            df = pd.read_csv(os.path.join(self.dataset_path, f + ".csv"), sep='\t')
            l = len(df.index)
            num_msg = int(l * self._seq_ratio)

            if len(df.index) < num_msg:
                num_msg = len(df.index)-1

            # Normalize also sec_since_start with regards to last value in the column
            conv_length =  df['sec_since_start'].iloc[-1]


            df = df.head(num_msg) # WE TAKE THE BEGINNING
            df = df.drop(columns=["event_id","message_id", "Unnamed: 0", "Unnamed: 0.1", "user_handle"])

            df = df.mean(axis = 0)
            x = df.values.astype(np.float32)
            x[-1] = x[-1] / conv_length

            if(np.isnan(x).any()):
                #print("The Array contain NaN values!!!!!!!!!!!!!!!!!!!!! IN PRE " + str(f))
                x = np.zeros(len(x))

            xt = torch.from_numpy(x).float()
            return xt

    def get_post_X(self, f):
            df = pd.read_csv(os.path.join(self.dataset_path, f + ".csv"), sep='\t')
            l = len(df.index)
            num_msg = int(l * self._seq_ratio)

            if len(df.index) < num_msg:
                num_msg = len(df.index)-1


            # Normalize also sec_since_start with regards to last value in the column
            conv_length =  df['sec_since_start'].iloc[-1]

            df = df.tail(num_msg) # WE TAKE THE END
            df = df.drop(columns=["event_id","message_id", "Unnamed: 0", "Unnamed: 0.1", "user_handle"])

            df = df.mean(axis = 0)

            x = df.values.astype(np.float32)
            x[-1] = x[-1] / conv_length

            if(np.isnan(x).any()):
                #print("The Array contain NaN values!!!!!!!!!!!!!!!!!!!!! IN POST " + str(f))
                x = np.zeros(len(x))


            xt = torch.from_numpy(x).float()
            return xt
    
    def get_post_eval(self, id):
        df = self.post_eval[self.post_eval["event_id"] == id]
        df = df.drop(columns=["event_id"])

        if self.negative:
            # We drop "I have a will to live column and average the other numbers"
            df = df.drop(columns=["Ik heb de wil om te leven"])
            y = df.mean(axis = 1)
            y = y.values.astype(np.float32)
        else:
            y = df['Ik heb de wil om te leven']
            y = y.values.astype(np.float32)

        yt = torch.tensor(y).float()
        return yt

    def get_pre_eval(self, id):
        df = self.pre_eval[self.pre_eval["event_id"] == id]
        df = df.drop(columns=["event_id"])

        if self.negative:
            # We drop "I have a will to live column and average the other numbers"
            df = df.drop(columns=["Ik heb de wil om te leven"])
            y = df.mean(axis = 1)
            y = y.values.astype(np.float32)
        else:
            y = df['Ik heb de wil om te leven']
            y = y.values.astype(np.float32)

        yt = torch.tensor(y).float()
        return yt

    def get_ids(self):

        post = os.path.join(self.dataset_path, "postchat_questions.tsv")
        pre = os.path.join(self.dataset_path, "prechat_questions.tsv")

        post_df = pd.read_csv(post, sep='\t')
        pre_df = pd.read_csv(pre, sep='\t')

        # get all ids which have a rating
        post_ids = post_df[post_df['Ik heb de neiging om mezelf te doden'] >= 0]['event_id']
        pre_ids = pre_df[pre_df['Ik heb de neiging om mezelf te doden'] >= 0]['event_id']

        self.post_ids = post_ids.tolist()
        self.pre_ids = pre_ids.tolist()

    def __len__(self):
        return len(self.pre_ids) + len(self.post_ids)


    def __getitem__(self, index):
        l_pre = len(self.pre_ids)
        l_post = len(self.post_ids)

        if index < l_pre:
            xt = self.get_pre_X(self.pre_ids[index])
            yt = self.get_pre_eval(self.pre_ids[index])
        else:
            index = index - l_pre
            xt = self.get_post_X(self.post_ids[index])
            yt = self.get_post_eval(self.post_ids[index])
        
        return xt, yt

