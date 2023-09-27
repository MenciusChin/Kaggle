"""
Train model and generate prediction
"""
# Import libraries
import pandas as pd
from preprocessing import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Global variable
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
TARGET = ["Survived"]


# Train model
def train(data, model):
    # Preprocessing
    data = preprocessing(data, FEATURES + TARGET)

    xdata = pd.get_dummies(data[FEATURES])
    ydata = data[TARGET]

    # Only options are top 3 models from notebook
    if model == "glm":
        glm = LogisticRegression()
        glm.fit(xdata, ydata)
        return glm
    elif model == "rf":
        rf = RandomForestClassifier(n_estimators=6)
        rf.fit(xdata, ydata)
        return rf
    else:
        gb = GradientBoostingClassifier(n_estimators=7, learning_rate=1.1)
        gb.fit(xdata, ydata)
        return gb


# Generate prediction
def predict(data, model):
    # Preprocessing
    data = preprocessing(data, FEATURES)

    xdata = pd.get_dummies(data)

    prediction = model.predict(xdata)

    prediction.to_csv("prediction.csv")


if __name__ == "__main__":
    model = train("data/train.csv", "rf")
    print(model)
    predict("data/test.csv", model)
