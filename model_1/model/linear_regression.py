import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR


class MLP(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''
    def __init__(self, input_len: int = 266, 
                    lr: float = 1e-2, 
                    lr_decay: float = 9e-1,
                    adam_betas: List[float] = [99e-2, 999e-3]
                ):

        super().__init__()

        # set up tensorboard
        self.__tb_sub = datetime.now().strftime("%m-%d-%Y_%H%M%S")
        self._tb_path = f"runs/Linear_Regression/{self.__tb_sub}"
        self._writer = SummaryWriter(self._tb_path)

        self.layers = nn.Sequential(
            nn.Linear(input_len, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)

        )

        self.__loss_func = nn.L1Loss()
        #self.__loss_func = nn.MSELoss() -> doesn't do well -> lot's of different outliers - error will explode

        self._optim = torch.optim.AdamW(self.parameters(), lr=lr)
        self.__sample_position = 0

        self._scheduler = ExponentialLR(self._optim, gamma=lr_decay)


        self._device = "cpu"
        if torch.cuda.is_available():
            self._device_name = torch.cuda.get_device_name(0)
            print(f"GPU acceleration available on {self._device_name}")

    def forward(self, x):
        '''
            Forward pass
        '''
        return self.layers(x)


    def use_device(self, device: str) -> None:
        self._device = device
        self.to(self._device)


    def save_to_default(self) -> None:
        """This method saves the current model state to the tensorboard 
        directory.
        """
        model_tag = datetime.now().strftime("%H%M%S")
        params = self.state_dict()
        torch.save(params, f"{self._tb_path}/model_{model_tag}.torch")

    def learn(self, loader: DataLoader, epochs: int = 1, validate: DataLoader = None, 
                test: DataLoader = None) -> None:
            dev_name = self._device_name if self._device == "cuda" else "CPU"
            print(f"Starting training on {dev_name}")
    
            # set the model into training mode
            self.train()

            for epoch in tqdm(range(epochs)):

                current_loss = 0.0
                # Iterate over the DataLoader for training data
                for x, y in loader:
                    x = x.to(self._device)
                    y = y.to(self._device)  

                    inputs = x
                    targets = y
                                        
                    # Zero the gradients
                    self._optim.zero_grad()
                    
                    # Perform forward pass
                    outputs = self(inputs)
                    #print(outputs)

                    # Compute loss
                    loss = self.__loss_func(outputs, targets)
                    
                    # Perform backward pass
                    loss.backward()

                    #torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                    
                    # Perform optimization
                    self._optim.step()
                    
                    self._writer.add_scalar(
                        "Train/loss", loss, self.__sample_position)
                    self._writer.add_scalar(
                        "Train/accuracy", ((7.0-loss)/7.0), self.__sample_position)

                    self.__sample_position += x.size(0)

                # if there is an adaptive learning rate (scheduler) available
                if self._scheduler:
                    self._scheduler.step()
                    lr = self._scheduler.get_last_lr()[0]
                    self._writer.add_scalar("Train/learning_rate", lr, epoch)


                if validate:
                    # set the model to eval mode, run validation and set to train mode again
                    self.eval()
                    accuracy = self.validate(validate)
                    self.train()


            # Process is complete.
            print('Training process has finished.')
            self._writer.flush()
            self.save_to_default()

    def validate(self, loader: DataLoader) -> None:
        losses = []
        accuracies = []
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self._device)
                y = y.to(self._device)  

                inputs = x
                targets = y
                                                    
                # Perform forward pass
                outputs = self(inputs)
                # Compute loss
                loss = self.__loss_func(outputs, targets)

                losses.append(loss.detach().cpu().item())
                accuracies.append((7.0 - loss.detach().cpu().item())/7.0)

       
        losses = np.array(losses)
        self._writer.add_scalar(
            "Validation/loss", np.mean(losses), self.__sample_position)

        accuracies = np.array(accuracies)
        self._writer.add_scalar(
            "Validation/accuracy", np.mean(accuracies), self.__sample_position)