"""
Python file for training and prediction
"""

import pandas as pd
import torch
from preprocessing import read_img, data_load
from model import GarmentClassifier, DigitNetwork, test, train

from torch import nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def model_train(model: nn.Module=GarmentClassifier()) -> nn.Module:
    # Read train data -> data loader
    imgs, labels = read_img("train.csv")
    imgtrain, imgtest, labeltrain, labeltest = train_test_split(imgs, labels, test_size=.3)
    train_dl = data_load(imgtrain, labeltrain)
    test_dl = data_load(imgtest, labeltest)

    model = GarmentClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    train_losses, train_counter, test_losses, test_counter, test_accuracies = train(
        model, optimizer, train_dl, test_dl, n_epochs=10, verbose=True
    )

    print("Final test loss:", test_losses[-1])
    print("Final test accuracies:", test_accuracies[-1])

    plt.plot(train_counter, train_losses, color="blue")
    plt.scatter(test_counter, test_losses, color="red")
    plt.legend(["Training loss", "Test loss"], loc="upper right")
    plt.xlabel("Number of training samples seen")
    plt.ylabel("Cross Entropy loss")
    plt.show()

    return model


def model_predict(model: nn.Module):
    # Read test data
    imgs, _ = read_img("test.csv")
    imgs = (torch.tensor(imgs)/255.0).unsqueeze(1)

    model.eval()
    output = model(imgs)
    pred = torch.argmax(output, dim=1)

    output = pd.DataFrame({"ImageId": range(1, len(pred)+1), "Label": pred.numpy()})
    return output


if __name__ == "__main__":
    trained = model_train()
    output = model_predict(model=trained)
    output.to_csv("prediction.csv", index=False)
    print(output.head())