"""
Python file for preprocess data
"""

# Libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


def read_train():
    data = pd.read_csv("data/train.csv")
    data.drop("id", axis=1, inplace=True)
    return data.drop("defects", axis=1), data["defects"]


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


def SMOTE_data(X, y):
    # Calculate Ratios
    counter = Counter(y.to_numpy().squeeze())
    total = dict(counter)[False] + dict(counter)[True]
    SMOTE_ratio = total // 2 / dict(counter)[False]
    RUS_ratio = 1

    # define pipeline
    over = SMOTE(sampling_strategy=SMOTE_ratio)
    under = RandomUnderSampler(sampling_strategy=RUS_ratio)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    # transform the dataset
    X, y = pipeline.fit_resample(X, y)

    return X, y