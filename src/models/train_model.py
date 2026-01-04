import yaml
import pandas as pd
import joblib
from pathlib import Path
from dvclive import Live

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def load_data():
    X = pd.read_csv("data/processed/X.csv")
    y = pd.read_csv("data/processed/y.csv").values.ravel()
    return X, y

def get_model(model_name, model_params):
    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**model_params)

    if model_name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(**model_params)

    if model_name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(**model_params)

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**model_params)

    if model_name == "svc":
        from sklearn.svm import SVC
        return SVC(**model_params)

    raise ValueError(f"Unsupported model: {model_name}")

def main():
    params = load_params()
    model_name = params["model"]["name"]
    model_params = params["models"][model_name]

    X, y = load_data()
    model = get_model(model_name, model_params)

    with Live(save_dvc_exp=True) as live:
        model.fit(X, y)

        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        live.log_param("model", model_name)
        for k, v in model_params.items():
            live.log_param(k, v)

if __name__ == "__main__":
    main()
