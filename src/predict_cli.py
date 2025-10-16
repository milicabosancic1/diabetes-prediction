import argparse, json
from typing import Dict
from src.infer import predict_single
from src.utils import FEATURES

_HINTS = {
    "Pregnancies": "broj trudnoća (0–17, ceo broj)",
    "Glucose": "glukoza (mg/dL, tipično 70–200; 0 = nepoznato)",
    "BloodPressure": "dijastolni pritisak (mmHg, 40–120; 0 = nepoznato)",
    "SkinThickness": "debljina kože – triceps (mm, 0–99; 0 = nepoznato)",
    "Insulin": "insulin (mu U/ml, 0–900; 0 = nepoznato)",
    "BMI": "BMI (kg/m^2, npr. 18.5–50; 0 = nepoznato)",
    "DiabetesPedigreeFunction": "DPF (porodična istorija, 0.08–2.5)",
    "Age": "godine (npr. 21–81)",
}

_EXAMPLE = {
    "Pregnancies": 2,
    "Glucose": 130,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 100,
    "BMI": 30.5,
    "DiabetesPedigreeFunction": 0.45,
    "Age": 35
}

def _prompt_yes_no(q: str) -> bool:
    while True:
        ans = input(q).strip().lower()
        if ans in ("da","d","yes","y"): return True
        if ans in ("ne","n","no"): return False
        print("Molim odgovori sa 'da' ili 'ne'.")

def _prompt_float(name: str, hint: str) -> float:
    while True:
        raw = input(f"  • {name} ({hint}): ").strip()
        try:
            return float(raw)
        except ValueError:
            print("  ↳ Unos mora biti broj (koristi tačku za decimalni zarez). Pokušaj ponovo.")

def _collect_interactive() -> Dict[str, float]:
    print("\nUnos podataka (za metrike gde je 0 dozvoljeno kao 'nepoznato' – slobodno unesi 0)")
    vals = {}
    for key in FEATURES:
        vals[key] = _prompt_float(key, _HINTS.get(key, "broj"))
    return vals

def main():
    parser = argparse.ArgumentParser(description="Interaktivna procena rizika (Diabetes) – CLI")
    parser.add_argument("--csv", default="data/diabetes.csv", help="putanja do CSV-a (Pima)")
    parser.add_argument("--family", choices=["NaiveBayes","KNN","LogReg","MLP"], default=None,
                        help="familija modela (ako izostaviš, bira se najbolja iz outputs/summary.json)")
    parser.add_argument("--json", type=str, default=None,
                        help="JSON sa poljima (Pregnancies, Glucose, ... Age). Ako je zadat, preskače se interaktivni unos.")
    args = parser.parse_args()

    print("\nDa li želiš da uneseš svoje podatke? (da/ne)")
    if args.json is None and _prompt_yes_no("> "):
        user_input = _collect_interactive()
    elif args.json:
        user_input = json.loads(args.json)
    else:
        print("\nNisi izabrala interaktivni unos. Evo primer JSON formata za --json:")
        print(json.dumps(_EXAMPLE, indent=2))
        return 0

    res = predict_single(user_input, csv_path=args.csv, family=args.family)

    label_txt = "POVIŠEN RIZIK" if res["label"] == 1 else "nema povišen rizik"
    print("\n=== Rezultat ===")
    print(f" Model:       {res['family']}")
    print(f" Prag (thr):  {res['threshold']:.4f}")
    print(f" Skor:        {res['score']:.4f}")
    print(f" Odluka:      {label_txt}")
    print("\nNapomena: edukativni alat; ne zamenjuje medicinski savet.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
