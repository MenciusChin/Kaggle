"""
Python script for data processing
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DigitDataset(Dataset):
    def __init__(self, images, labels):
        self.images = (torch.tensor(images)/255.0).unsqueeze(1)     # Match Conv2D input
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def read_img(file_name:str) -> (np.ndarray, np.ndarray):
    data = pd.read_csv("data/" + file_name)
    if "label" in data.columns:
        labels = data["label"].to_numpy()
        imgs = (data.iloc[:, 1:].to_numpy()).reshape((len(labels),28,28)).astype("float32")
    else:
        labels = None
        imgs = (data.to_numpy()).reshape((len(data),28,28)).astype("float32")
    return imgs, labels


def data_load(imgs: np.ndarray, labels: np.ndarray, batch_size: int=32) -> DataLoader:
    ds = DigitDataset(imgs, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    imgs, labels = read_img("test.csv")
    print("Test file without labels:", not labels)

    imgs, labels = read_img("train.csv")
    train_dl = data_load(imgs, labels, 16)
    # Display image and label.
    train_features, train_labels = next(iter(train_dl))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    print(f"Label: {label}")
    plt.imshow(img, cmap="gray")
    plt.show()
    