import os
import sys

from src.plotting import generate_all_plots
from src.train_eval import train_and_evaluate_all

BASE_DIR = os.path.dirname(__file__)

def resolve_csv_path() -> str:
    data_csv = os.path.join(BASE_DIR, "data", "diabetes.csv")
    if os.path.isfile(data_csv):
        return data_csv

    # fallback na sample ako postoji
    sample_csv = os.path.join(BASE_DIR, "data", "diabetes_sample.csv")
    if os.path.isfile(sample_csv):
        print("⚠️  Nema data/diabetes.csv – koristim data/diabetes_sample.csv")
        return sample_csv

    raise FileNotFoundError(
        "Nisam našao ni data/diabetes.csv ni data/diabetes_sample.csv. "
        "Dodaj CSV u folder data/ pa pokreni ponovo."
    )

def main() -> int:
    try:
        csv_path = resolve_csv_path()
        out_dir = os.path.join(BASE_DIR, "outputs")
        train_and_evaluate_all(csv_path=csv_path, out_dir=out_dir, seed=42)
        generate_all_plots(outputs_dir=os.path.join(BASE_DIR, 'outputs'))
        print("✅ Gotovo. Rezultati su u 'outputs/'.")
        return 0
    except Exception as e:
        print(f"❌ Greška: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
