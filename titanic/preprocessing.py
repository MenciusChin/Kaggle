"""
Python script for data preprocessing
"""
# Import Libraries
import pandas as pd


# Function for read in and preprocess the raw data
def preprocessing(path, columns):
    # Read data
    data = pd.read_csv(path)
    data = data[columns]

    # Missing Value imputation for titanic dataset
    data["Embarked"].fillna("S", inplace=True)
    data["Age"].fillna(data["Age"].median(), inplace=True)

    return data


if __name__ == "__main__":
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    target = ["Survived"]
    data = preprocessing("data/train.csv", features + target)
    print(data.head())
