import os, sys, urllib.request, pandas as pd

CANDIDATE_URLS = [
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data",
]
COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    try: return raw.decode("utf-8")
    except UnicodeDecodeError: return raw.decode("latin-1")


def try_read_csv_from_text(txt: str) -> pd.DataFrame:
    import io
    buf = io.StringIO(txt)
    try:
        df = pd.read_csv(buf)
        if df.shape[1] != 9:
            buf.seek(0)
            df = pd.read_csv(buf, header=None, names=COLUMNS)
    except Exception:
        buf.seek(0)
        df = pd.read_csv(buf, header=None, names=COLUMNS)
    if df.shape[1] != 9:
        raise ValueError("Unexpected number of columns")
    df.columns = COLUMNS
    return df


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    dest = os.path.join(data_dir, "diabetes.csv")
    last_err=None
    for url in CANDIDATE_URLS:
        print("Trying:", url)
        try:
            txt = fetch_text(url)
            df = try_read_csv_from_text(txt)
            df.to_csv(dest, index=False)
            print("Saved:", dest, "rows:", len(df))
            return
        except Exception as e:
            print("Failed:", e); last_err=e
    print("Could not download dataset. Last error:", last_err); sys.exit(1)


if __name__ == "__main__":
    main()
