"""
Python file for modeling
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

loss_fn=torch.nn.CrossEntropyLoss()
batch_size_train = 32
batch_size_test = 1000


# PyTorch models inherit from torch.nn.Module
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Activation Tanh
class DigitNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2dstack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5)),
            nn.Tanh(),
            nn.MaxPool2d((2,2)),
            nn.Tanh(),
            nn.Conv2d(6, 16, (5,5), groups=2),
            nn.Tanh(),
            nn.MaxPool2d((2,2)),
            nn.Tanh()
        )
        self.flatten = nn.Flatten()
        self.t1 = nn.Tanh()
        self.linear = nn.LazyLinear(100)
        self.t2 = nn.Tanh()
        self.output = nn.Linear(100, 10)


    def forward(self, x):
        x = self.conv2dstack(x)
        x = self.flatten(x)
        x = self.t1(x)
        x = self.linear(x)
        x = self.t2(x)
        logits = self.output(x)
        return logits
    

def test(model, loader):
    """Test a network model on the test data.
    
    model: a `nn.Module` object representing the neural network to test
    loader: a `DataLoader` object containing the testing data, potentially in batches
    
    Returns a tuple (test_loss, accuracy), where test_loss is the average loss on the
    test data, and accuracy is the accuracy rate (out of 100%) of the predictions.
    """
    
    # Put the network in evaluation mode
    model.eval()
    
    test_loss = 0
    correct = 0
    
    # don't track gradients during this calculation
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            
            # calculate the loss on this batch
            test_loss += loss_fn(output, target)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    # get the overall loss across all batches
    test_loss /= len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)
    
    return test_loss, accuracy
    

def train(model, optimizer, train_loader, test_loader, n_epochs=5, log_interval=50, verbose=False):
    """Train a network model with a particular optimizer.
    
    model: a `nn.Module` object representing the neural network to train
    optimizer: a PyTorch optimizer, such as SGD
    train_loader: a `DataLoader` object containing the training data, potentially in batches
    test_loader: the same, but for the testing data
    n_epochs: number of epochs of training to run
    log_interval: after how many batches should record our progress?
    verbose: print the progress after each log_interval steps?
    
    Returns a tuple (train_losses, counter, test_losses). train_losses is a list of 
    losses after every log_interval training steps; train_counter is a list giving the number
    of training observations seen by the training process by each of those steps; test_losses
    gives the test-set loss before each epoch and after the final epoch; test_counter gives
    the number of training observations seen at each epoch.
    """
    
    # Storage
    train_losses = []
    test_losses = []
    train_counter = []
    test_counter = []
    test_accuracies = []
    
    samples_seen = 0
    
    for epoch in range(n_epochs):
        # test at the beginning of each epoch
        test_loss, _ = test(model, test_loader)
        test_losses.append(test_loss)
        test_counter.append(samples_seen)
        
        model.train() # put the network in training mode

        for batch_idx, (data, target) in enumerate(train_loader):
            # Reset all gradients
            optimizer.zero_grad()
    
            # Obtain the model's predictions with the data
            predictions = model(data)
    
            # Get the loss, using the known Ys
            loss = loss_fn(predictions, target)
    
            # Calculate the gradients with backpropagation
            loss.backward()
    
            # Perform one gradient descent step
            optimizer.step()

            # Count how much data we've seen
            samples_seen += batch_size_train
            
            # Every log_interval steps, print out the progress
            if batch_idx % log_interval == 0:
                if verbose:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader), loss.item()))
                
                train_losses.append(loss.item())
                train_counter.append(samples_seen)
                

    # test after the final epoch
    test_loss, test_accuracy = test(model, test_loader)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    test_counter.append(samples_seen)
                
    return train_losses, train_counter, test_losses, test_counter, test_accuracies
