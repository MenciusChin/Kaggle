"""
Python file for preprocess data
"""

# Libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# Read train data
def read_train(original=False):
    data = pd.read_csv("data/train.csv")
    data.drop("id", axis=1, inplace=True)

    if original:
        # For original data it contains "?", we treat which as Missing Values
        original = pd.read_csv("data/jm1.csv")
        TAR = ["uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount"]
        for col in TAR:
            original[col].replace("?", value=0, inplace=True)
            original[col] = pd.to_numeric(original[col])
        data = pd.concat([data, original])

    return data.drop("defects", axis=1), data["defects"]


# Read test data
def read_test():
    data = pd.read_csv("data/test.csv")
    id = data["id"].values
    data.drop("id", axis=1, inplace=True)
    return data, id


# Check distributions
def plot_dist(data, log_scale=False):
    fig, axes = plt.subplots(5, 5, figsize=(24, 20))
    cols = data.columns
    for i in range(5):
        for j in range(5):
            idx = i*5+j
            if idx > len(cols)-1:
                return
            sns.histplot(ax=axes[i, j], data=data[cols[idx]] + 0.0001, kde=True, bins=15, log_scale=log_scale)
    plt.show()


# SMOTE sampling to have 2:1 Majority and Minority classes
def SMOTE_data(X, y):
    # Calculate Ratios
    counter = Counter(y.to_numpy().squeeze())
    total = dict(counter)[False] + dict(counter)[True]
    SMOTE_ratio = 1 / 2
    RUS_ratio = 2 / 3

    # define pipeline
    over = SMOTE(sampling_strategy=SMOTE_ratio)
    under = RandomUnderSampler(sampling_strategy=RUS_ratio)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    # transform the dataset
    X, y = pipeline.fit_resample(X, y)

    return X, y

def transform_X(X):
    ss = StandardScaler()
    return ss.fit_transform(np.log(X + .001))


if __name__ == "__main__":
    X, y = read_train()
    X, y = SMOTE_data(X, y)
    
