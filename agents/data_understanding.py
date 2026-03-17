import pandas as pd

def data_understanding(file_path):
    df = pd.read_csv(file_path)

    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nData types:\n", df.dtypes)

if __name__ == "__main__":
    data_understanding("your_data.csv")
