"""
Python file for read in data
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# GLOBAL VALUE FOR COLUMNS
CON = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CAT = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
TAR = ["Transported"]


def preprocessing(file_name):
    data = pd.read_csv("data/" + file_name)

    # Missing value imputation
    for col in CON:
        data[col].fillna(data[col].median(), inplace=True)

    # For categorical value we encode
    encode = pd.get_dummies(data[CAT])

    X = data[CON].join(encode)
    Y = data[TAR]*1 if "Transported" in data.columns else data["PassengerId"]

    ss = StandardScaler()
    X = ss.fit_transform(X)

    return X, Y



if __name__ == "__main__":
    x, y = preprocessing("test.csv")
    print(y)