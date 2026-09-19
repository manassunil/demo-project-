from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = Path("ml/datasets/depression_reddit_cleaned_ds.csv")
OUTPUT_DIR = Path("ml/datasets/processed")


def prepare_dataset():

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Original rows: {len(df)}")

    # Keep only the columns we need
    df = df[["text", "labels"]]

    # Remove missing values
    df = df.dropna(subset=["text", "labels"])

    # Convert text to string
    df["text"] = df["text"].astype(str).str.strip()

    # Remove empty text
    df = df[df["text"] != ""]

    # Remove duplicate texts
    before_duplicates = len(df)

    df = df.drop_duplicates(subset=["text"])

    duplicates_removed = before_duplicates - len(df)

    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Rows after cleaning: {len(df)}")

    # Make sure labels are integers
    df["labels"] = df["labels"].astype(int)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 80% training, 20% temporary
    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["labels"]
    )

    # Split temporary 50/50
    # Result: 10% validation, 10% test
    validation_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["labels"]
    )

    # Save files
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)

    validation_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)

    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\nDataset preparation complete!")

    print(f"Training samples:   {len(train_df)}")
    print(f"Validation samples: {len(validation_df)}")
    print(f"Test samples:       {len(test_df)}")

    print("\nTraining labels:")
    print(train_df["labels"].value_counts())

    print("\nValidation labels:")
    print(validation_df["labels"].value_counts())

    print("\nTest labels:")
    print(test_df["labels"].value_counts())


if __name__ == "__main__":
    prepare_dataset()
