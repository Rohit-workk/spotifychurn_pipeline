from pathlib import Path
import pandas as pd

def main():
    raw_path = Path("data/raw/spotify_churn_dataset.csv")
    interim_path = Path("data/interim")
    interim_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    df.to_csv(interim_path / "dataset.csv", index=False)

if __name__ == "__main__":
    main()
