"""
Titanic prediction script
"""

import sys
from prediction import train, predict


if __name__ == "__main__":
    test = "data/" + sys.argv[1]
    model = sys.argv[2]

    if model not in {"glm", "rf", "gb"}:
        raise ValueError("Not valid option for model")

    model = train("data/train.csv", model)
    predict(test, model)
