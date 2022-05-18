
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler
import scipy.io
import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, d_type: str = "train", normalize: bool = True, bounds: Tuple[int] = (0, 1),
                future_steps: int= 1, sequence_length: int = 1, precision: np.dtype = np.float32):
        super(Dataset, self).__init__()

        self._precision = precision
        self._seq = sequence_length
        self._f_seq = future_steps

        # load the dataset specified
        self._file = f"./data/{d_type}.mat"
        self._mat = scipy.io.loadmat(self._file).get(f"X{d_type}")
        self._mat = self._mat.astype(self._precision)

        # normalize the dataset between values of o to 1
        self._scaler = None
        if normalize:
            self._scaler = MinMaxScaler(feature_range=bounds)
            self._scaler.fit(self._mat)
            self._mat = self. _scaler.transform(self._mat)

    @property
    def sample_size(self) -> int:
        return 1
    
    def scale_back(self, data):
        data = np.array(data, dtype=self._precision)
        return self._scaler.inverse_transform(data)

    def __len__(self):
        return max(1, len(self._mat) - self._f_seq - self._seq)

    def __getitem__(self, index):
        #X = np.zeros((self._seq, 1), dtype=self._precision)
        #y = np.zeros((self._f_seq, 1), dtype=self._precision)
        X = self._mat[index:self._seq + index]
        y = self._mat[self._seq + index:self._seq + index + self._f_seq]
        return X, y

        if index == 0:
            return X, y
        
        if index > len(self):
            return None, None

        # calculate indecies for adding to the sequence
        y_start = index - self._f_seq
        y_end = index
        x_start = y_start - self._seq
        x_end = y_start

        # define the label sequence y
        # occurs also if x_end < 0
        if y_start <= 0:
            y[-self._f_seq - index:] = self._mat[index:self._f_seq + index]
            return X, y

        y = self._mat[index - self._f_seq:index]
        y = y.astype(self._precision)

        # define the training data X
        if x_start < 0:
            X[self._seq - x_end:] = self._mat[:x_end]
            return X, y

        X = self._mat[x_start:x_end]
        return X, y
