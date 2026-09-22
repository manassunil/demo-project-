from pathlib import Path

import pandas as pd


REDDIT_DIR = Path("ML/datasets/processed")
MHC_DIR = Path("ML/datasets/mental_health_corpus/processed")

OUTPUT_DIR = Path("ML/datasets/combined")


def load_split(directory: Path, filename: str, source: str):
    df = pd.read_csv(directory / filename)

    df = df[["text", "labels"]].copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["labels"] = df["labels"].astype(int)

    df = df[df["text"] != ""]

    # Keep source information for evaluation/debugging.
    df["source"] = source

    return df


def combine():
    print("Loading datasets...")

    reddit_train = load_split(
        REDDIT_DIR,
        "train.csv",
        "reddit",
    )

    reddit_validation = load_split(
        REDDIT_DIR,
        "validation.csv",
        "reddit",
    )

    reddit_test = load_split(
        REDDIT_DIR,
        "test.csv",
        "reddit",
    )

    mhc_train = load_split(
        MHC_DIR,
        "train.csv",
        "mhc",
    )

    mhc_validation = load_split(
        MHC_DIR,
        "validation.csv",
        "mhc",
    )

    mhc_test = load_split(
        MHC_DIR,
        "test.csv",
        "mhc",
    )

    # Combine matching splits.
    train_df = pd.concat(
        [reddit_train, mhc_train],
        ignore_index=True,
    )

    validation_df = pd.concat(
        [reddit_validation, mhc_validation],
        ignore_index=True,
    )

    test_df = pd.concat(
        [reddit_test, mhc_test],
        ignore_index=True,
    )

    # Remove any duplicates within the combined splits.
    train_df = train_df.drop_duplicates(subset=["text"])

    validation_df = validation_df.drop_duplicates(subset=["text"])

    test_df = test_df.drop_duplicates(subset=["text"])

    # Shuffle while preserving reproducibility.
    train_df = train_df.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    validation_df = validation_df.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    test_df = test_df.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Training files contain only the columns
    # needed by DistilBERT.
    train_df[["text", "labels"]].to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )

    validation_df[["text", "labels"]].to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False,
    )

    test_df[["text", "labels"]].to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
    )

    # Keep source information separately for analysis.
    test_df.to_csv(
        OUTPUT_DIR / "test_with_source.csv",
        index=False,
    )

    print("\nCombined dataset created.")

    print("\nRow counts:")
    print(f"Train:      {len(train_df)}")
    print(f"Validation: {len(validation_df)}")
    print(f"Test:       {len(test_df)}")

    print("\nTraining label distribution:")
    print(train_df["labels"].value_counts().sort_index())

    print("\nTraining source distribution:")
    print(train_df["source"].value_counts())

    print("\nValidation source distribution:")
    print(validation_df["source"].value_counts())

    print("\nTest source distribution:")
    print(test_df["source"].value_counts())

    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    combine()
