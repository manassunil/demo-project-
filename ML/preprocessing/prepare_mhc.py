from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_FILE = Path("ML/datasets/mental_health_corpus/mental_health.csv")
OUTPUT_DIR = Path("ML/datasets/mental_health_corpus/processed")


def prepare_dataset():
    print("Loading Mental Health Corpus...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Original rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # Keep only the columns we need
    df = df[["text", "label"]].copy()

    # Rename label to match our existing project format
    df = df.rename(columns={"label": "labels"})

    # Remove missing values
    df = df.dropna(subset=["text", "labels"])

    # Clean text
    df["text"] = df["text"].astype(str).str.strip()

    df = df[df["text"] != ""]

    # Make labels integers
    df["labels"] = df["labels"].astype(int)

    # Remove duplicate texts
    before_duplicates = len(df)

    df = df.drop_duplicates(subset=["text"])

    duplicates_removed = before_duplicates - len(df)

    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Rows after cleaning: {len(df)}")

    print("\nLabel distribution:")
    print(df["labels"].value_counts())

    # First split:
    # 80% train
    # 20% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["labels"],
    )

    # Split temporary set equally:
    # 10% validation
    # 10% test
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["labels"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )

    validation_df.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False,
    )

    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
    )

    print("\nSaved files:")

    print(f"Train:      {len(train_df)}")
    print(f"Validation: {len(validation_df)}")
    print(f"Test:       {len(test_df)}")

    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    prepare_dataset()
