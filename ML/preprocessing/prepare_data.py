import pandas as pd


DATASET_PATH = "ml/datasets/depression_dataset.csv"


def load_dataset():
    df = pd.read_csv(DATASET_PATH)

    print("Dataset loaded!")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    return df


if __name__ == "__main__":
    load_dataset()
