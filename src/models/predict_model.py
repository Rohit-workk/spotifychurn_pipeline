import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score

def main():
    X = pd.read_csv("data/processed/X.csv")
    y = pd.read_csv("data/processed/y.csv")

    model = joblib.load("models/model.pkl")
    preds = model.predict(X)

    acc = accuracy_score(y, preds)

    with open("metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f)

if __name__ == "__main__":
    main()
