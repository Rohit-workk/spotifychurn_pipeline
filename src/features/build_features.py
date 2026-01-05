from pathlib import Path
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
import joblib

def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()

    df = pd.read_csv("data/interim/dataset.csv")

    cat_cols = params["features"]["categorical_columns"]

    df = pd.get_dummies(
        df,
        columns=cat_cols,
        drop_first=True,
        dtype=int
    )

    X = df.drop(columns=["is_churned"])
    y = df["is_churned"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    Path("data/processed").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    pd.DataFrame(X_scaled).to_csv("data/processed/X.csv", index=False)
    y.to_csv("data/processed/y.csv", index=False)

    joblib.dump(scaler, "models/scaler.pkl")

if __name__ == "__main__":
    main()
