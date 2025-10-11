import os

BASE_DIR = os.path.dirname(__file__)
def main():
    data_csv = os.path.join(BASE_DIR,"data","diabetes.csv")
    if not os.path.isfile(data_csv):
        print("Nema data/diabetes.csv — koristi se sample data/diabetes_sample.csv")
        data_csv = os.path.join(BASE_DIR,"data","diabetes_sample.csv")

if __name__ == "__main__":
    main()