from datasets import load_dataset
from pathlib import Path


DATASET_NAME = "mrjunos/depression-reddit-cleaned"
OUTPUT_DIR = Path("ml/datasets")


def download_dataset():
    print("Downloading dataset...")

    dataset = load_dataset(DATASET_NAME)

    print("\nDataset downloaded successfully!")
    print(dataset)

    train_data = dataset["train"]

    print("\nNumber of rows:", len(train_data))
    print("Columns:", train_data.column_names)

    print("\nFirst example:")
    print(train_data[0])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "depression_dataset.csv"

    train_data.to_csv(output_file, index=False)

    print("\nCSV saved to:")
    print(output_file)


if __name__ == "__main__":
    download_dataset()
