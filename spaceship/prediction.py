"""
Python file for model training and prediction
"""

import numpy as np
import pandas as pd

# Sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold

# Models
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from preprocessing import preprocessing

CLASSIFIERS = {
    "CatBoost" : CatBoostClassifier(learning_rate=0.15, max_depth=4, n_estimators=100, verbose=False, random_state=0),
}


def train(x, y):
    xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, test_size=.3)
    cb = CLASSIFIERS["CatBoost"]
    
    cb.fit(xtrain, ytrain)
    print(cb.score(xvalid, yvalid))

    return cb






if __name__ == "__main__":
    xtrain, ytrain = preprocessing("train.csv")
    model = train(xtrain, ytrain)

    xtest, ids = preprocessing("test.csv")
    preds = model.predict(xtest)

    output = pd.DataFrame({"PassengerId": ids, "Transported": preds})
    output["Transported"] = output["Transported"].astype(bool)
    output.to_csv("predictions.csv", index=False)
