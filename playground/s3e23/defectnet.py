"""
Python file for DefectNetwork and DefectDataset
"""

# Libraries
import numpy as np
import pandas as pd

from preprocessing import read_train, read_test, SMOTE_data, plot_dist, transform_X

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

loss_fn = nn.BCELoss()
batch_size_train=128

# Create dataset object
class DefectDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.astype(int).values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class DefectNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(21, 64)
        self.relu = nn.ReLU()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
        )
        self.output = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.linear_tanh_stack(x)
        x = self.output(x)
        logits = self.sigmoid(x)

        return logits
    

def test(model, loader, roc=False):
    """Test a network model on the test data.
    
    model: a `nn.Module` object representing the neural network to test
    loader: a `DataLoader` object containing the testing data, potentially in batches
    
    Returns a tuple (test_loss, accuracy), where test_loss is the average loss on the
    test data, and accuracy is the accuracy rate (out of 100%) of the predictions.
    """
    
    # Put the network in evaluation mode
    model.eval()
    
    test_loss = 0
    roc_scores = []
    
    # don't track gradients during this calculation
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            target = target.unsqueeze(1)
            
            # calculate the loss on this batch
            test_loss += loss_fn(output, target)
            if roc:
                pred = output.detach().numpy()
                roc_scores.append(roc_auc_score(target, pred))

    # get the overall loss across all batches
    test_loss /= len(loader.dataset)

    if roc:
        roc_score = np.mean(roc_scores)
        return test_loss, roc_score
    else:
        return test_loss
    

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
    
    samples_seen = 0
    
    for epoch in range(n_epochs):
        # test at the beginning of each epoch
        test_loss = test(model, test_loader)
        test_losses.append(test_loss)
        test_counter.append(samples_seen)
        
        model.train() # put the network in training mode

        for batch_idx, (data, target) in enumerate(train_loader):
            # Reset all gradients
            optimizer.zero_grad()
    
            # Obtain the model's predictions with the data
            predictions = model(data)
            target = target.unsqueeze(1)
    
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
    test_loss, test_roc_score = test(model, test_loader, roc=True)
    test_losses.append(test_loss)
    test_counter.append(samples_seen)
                
    return train_losses, train_counter, test_losses, test_counter, test_roc_score


def defectnet_pipeline(n_epochs=10, verbose=False, graph=False):
    # Data preprocess
    X, y = read_train()
    X, y = SMOTE_data(X, y)
    fit_X = transform_X(X)
    xtrain, xtest, ytrain, ytest = train_test_split(fit_X, y, test_size=.3)

    train_ds = DefectDataset(xtrain, ytrain)
    test_ds = DefectDataset(xtest, ytest)

    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=True)

    model = DefectNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses, train_counter, test_losses, test_counter, test_roc_score = train(
        model, optimizer, train_dl, test_dl, n_epochs=n_epochs, verbose=verbose
    )

    if verbose:
        print("Final test loss:", test_losses[-1].item())
        print("Final test roc score:", test_roc_score)
    
    if graph:
        plt.plot(train_counter, train_losses, color="blue")
        plt.scatter(test_counter, test_losses, color="red")
        plt.legend(["Training loss", "Test loss"], loc="upper right")
        plt.xlabel("Number of training samples seen")
        plt.ylabel("BCEloss")
        plt.show()
    
    # Return trained model
    return model


def prediction(model, write=False):
    X, id = read_test()
    fit_X = transform_X(X)
    input = torch.tensor(fit_X, dtype=torch.float32)

    pred = model(input).detach().numpy()
    output = [int(p) for p in pred >= 0.5]

    file = pd.DataFrame({"id": id, "defects": output})

    if write:
        file.to_csv("submission.csv")
    
    return file




if __name__ == "__main__":
    model = defectnet_pipeline(verbose=True)
    pred = prediction(model)
    print(pred.head())




