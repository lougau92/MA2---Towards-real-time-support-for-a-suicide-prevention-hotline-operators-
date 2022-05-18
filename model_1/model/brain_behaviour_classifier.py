from datetime import datetime
from typing import List, Tuple
from unicodedata import bidirectional
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from submodules.TimeSeriesDL.model.base_model import BaseModel
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm


# definitely a CNN again, maybe also LSTM
# reshape the input to (8, 31) and apply small filter (3x3 or 5x5)


class BrainBehaviourClassifier(BaseModel):
    def __init__(self, lr: float = 1e-3, lr_decay: float = 9e-1, 
                 adam_betas: List[float] = [99e-2, 999e-3]) -> None:
        # set up tensorboard
        self.__tb_sub = datetime.now().strftime("%m-%d-%Y_%H%M%S")
        self._tb_path = f"runs/BrainBehaviourClassifier/{self.__tb_sub}"
        self._writer = SummaryWriter(self._tb_path)

        super(BrainBehaviourClassifier, self).__init__()

        # first part of the neural network is CNN only which tries to predict
        # one of the 4 classes without taking a sequence into account
        self.__cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, 7, 1, 0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(1),
            torch.nn.Dropout(0.3),
            torch.nn.Conv2d(1, 2, 7, 1, 0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(2),
            torch.nn.Dropout(0.3),
            torch.nn.Conv2d(2, 4, 7, 1, 0),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU()
        )

        # second part of the neural network is a LSTM which takes the previous
        # output as an input and tries to predict one of the 4 classes with
        # taking the sequence into account
        self.__lstm = torch.nn.LSTM(24, 128, num_layers=4, dropout=0.3, bidirectional=False, batch_first=True)
        self.__final_dense = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 4)
        )

        self.__loss_func = torch.nn.CrossEntropyLoss()
        self._optim = torch.optim.AdamW(self.parameters(), lr=lr, betas=adam_betas)
        self._scheduler = ExponentialLR(self._optim, gamma=lr_decay)
        self.__sample_position = 0

    def learn(self, loader: DataLoader, epochs: int = 1, validate: DataLoader = None, 
              test: DataLoader = None) -> None:
        dev_name = self._device_name if self._device == "cuda" else "CPU"
        print(f"Starting training on {dev_name}")

        for epoch in tqdm(range(epochs)):
            self.train()
            
            for x, y in loader:
                x = x.to(self._device)
                y = y.to(self._device)

                _y = self(x)
                y = torch.flatten(y)
                loss = self.__loss_func(_y, y)

                self._optim.zero_grad()
                loss.backward()
                self._optim.step()

                self._writer.add_scalar(
                    "Train/loss", loss, self.__sample_position)
                self._writer.add_scalar(
                    "Train/accuracy", self._single_accuracy(_y, y), self.__sample_position)

                self.__sample_position += x.size(0)

            # if there is an adaptive learning rate (scheduler) available
            if self._scheduler:
                self._scheduler.step()
                lr = self._scheduler.get_last_lr()[0]
                self._writer.add_scalar("Train/learning_rate", lr, epoch)
            
            self.eval()
            if validate:
                self.validate(validate)
            
            if test:
                self.accuracy(test)

            self._writer.flush()

    def validate(self, loader: DataLoader) -> None:
        losses = []
        accuracies = []
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self._device)
                y = y.to(self._device)

                _y = self(x)
                y = torch.flatten(y)
                loss = self.__loss_func(_y, y)
                losses.append(loss.detach().cpu().item())
                accuracies.append(self._single_accuracy(_y, y))

        losses = np.array(losses)
        self._writer.add_scalar(
            "Validation/loss", np.mean(losses), self.__sample_position)

        accuracies = np.array(accuracies)
        self._writer.add_scalar(
            "Validation/accuracy", np.mean(accuracies), self.__sample_position)

    def _single_accuracy(self, pred_y, test_y) -> float:
        y_pred_softmax = torch.log_softmax(pred_y, dim = 1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    

        correct_pred = (y_pred_tags == torch.flatten(test_y)).float()
        acc = correct_pred.sum() / len(correct_pred)
        acc = torch.round(acc * 100)
        return acc.detach().cpu().item()

    def accuracy(self, loader: DataLoader) -> float:
        self.eval()
        accuracies = 0
        total = 0
        losses = []
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self._device)
                y = y.to(self._device)

                _y = self(x)
                loss = self.__loss_func(_y, torch.flatten(y))
                losses.append(loss.detach().cpu().item())

                acc = self._single_accuracy(_y, y)
                accuracies += acc
                total += 1
            
    def downsample(self, matrix):
        # Downsample the matrix by just deleting columns on a uniform distribution
        shape = matrix.shape
        matrix_time_steps = shape[1]

        if self.downsample_by > 1.0:
            print(f'downsample_by must be between 0 and 1!')
            return

        # We only want self.time_steps number of time steps after deletion
        # num _of_samples represents the amount of samples we will delete
        num_of_samples = matrix_time_steps - self.time_steps
        
        to_delete = np.linspace(0,matrix_time_steps-1,num_of_samples, dtype=int)
        matrix  = np.delete(matrix, to_delete, axis=1)

        return matrix
        losses = np.array(losses)
        self._writer.add_scalar(
            "Train/loss", np.mean(losses), self.__sample_position)

        acc_mean = accuracies / total
        self._writer.add_scalar(
            "Train/accuracy", acc_mean, self.__sample_position)

        return acc_mean
    
    def forward(self, x: torch.tensor) -> Tuple[torch.tensor]:
        batch_size, seq_size, dim_1, dim_2 = x.shape

        # for the cnn the batch and sequence counts as one, so combine both
        # view preserves the tensors the original order of the tensor
        x = x.view(-1, dim_1, dim_2)
        x = torch.unsqueeze(x, 1)
        x = self.__cnn(x)

        # restore the original shape
        # dim_1: channels 
        # dim_2: width
        # dim_3: height
        _, dim_1, dim_2, dim_3 = x.shape
        x = x.view(batch_size, seq_size, dim_1, dim_2, dim_3)

        # flatten the channels, width and height
        x = torch.flatten(x, 2, -1)
        
        # forward passing the CNN output into the LSTM
        x, _ = self.__lstm(x)

        # lstm output shape is batch_size, sequence_length, lstm_hidden_dimension
        x = x[:, -1, :]

        x = self.__final_dense(x)
        x = torch.softmax(x, dim=-1)

        # returns output of CNN and LSTM
        return x