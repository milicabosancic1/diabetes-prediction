import pandas as pd


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    y = df["Outcome"].astype(int).to_numpy()
    X = df.drop(columns=["Outcome"])
    return X, y
