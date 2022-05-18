from pkgutil import get_data
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import scipy.io
import torch
import numpy as np
import os
import pandas as pd


#from .utils import get_dataset_matrix, get_file_label, get_meshes


class CDS_Dataset(torch.utils.data.Dataset):
    def __init__(self, d_type: str = "train", sequence_ratio: int = 10,
                     dataset_path: str = "./data"):
        super(CDS_Dataset, self).__init__()

        self._seq_ratio = sequence_ratio

        self.dataset_path  = dataset_path

        # We open prechat and postchat questions files and extract out the 
        # conversation id's that have a rating

        self.all_files = os.listdir(self.dataset_path)


    
        self.pre_ids = list()
        self.post_ids = list()
        self.get_ids()


        self.pre_eval = pd.read_csv(os.path.join(self.dataset_path, "prechat_questions.tsv"), sep='\t')
        self.post_eval = pd.read_csv(os.path.join(self.dataset_path, "postchat_questions.tsv"), sep='\t')

        self.X = []
        self.y = []

        #self.prepare_X()

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

            df = df.head(num_msg) # WE TAKE THE BEGINNING
            df = df.drop(columns=["event_id","message_id", "Unnamed: 0", "Unnamed: 0.1", "user_handle"])

            df = df.mean(axis = 0)

            x = df.values.astype(np.float32)
            xt = torch.tensor(x).float()
            return xt

    def get_post_X(self, f):
            df = pd.read_csv(os.path.join(self.dataset_path, f + ".csv"), sep='\t')
            l = len(df.index)
            num_msg = int(l * self._seq_ratio)

            df = df.tail(num_msg) # WE TAKE THE BEGINNING
            df = df.drop(columns=["event_id","message_id", "Unnamed: 0", "Unnamed: 0.1", "user_handle"])

            df = df.mean(axis = 0)

            x = df.values.astype(np.float32)
            xt = torch.tensor(x).float()
            return xt
    
    def get_post_eval(self, id):
        df = self.post_eval[self.post_eval["event_id"] == id]
        df = df.drop(columns=["event_id"])

        y = df.mean(axis = 1)
        y = y.values.astype(np.float32)

        yt = torch.tensor(y).float()
        return yt

    def get_pre_eval(self, id):
        df = self.pre_eval[self.pre_eval["event_id"] == id]
        df = df.drop(columns=["event_id"])

        y = df.mean(axis = 1)
        y = y.values.astype(np.float32)

        yt = torch.tensor(y).float()
        return yt

    def get_ids(self):

        post = os.path.join(self.dataset_path, "postchat_questions.tsv")
        pre = os.path.join(self.dataset_path, "prechat_questions.tsv")

        post_df = pd.read_csv(post, sep='\t')
        pre_df = pd.read_csv(pre, sep='\t')

        # get all ids which have a rating
        post_ids = post_df[post_df['Ik heb de neiging om mezelf te doden'] > 0]['event_id']
        pre_ids = pre_df[pre_df['Ik heb de neiging om mezelf te doden'] > 0]['event_id']

        self.post_ids = post_ids.tolist()
        self.pre_ids = pre_ids.tolist()

        #print(f"Len of post ids : {len(self.post_ids)}")
        #print(f"Len of pre ids : {len(self.pre_ids)}")


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


