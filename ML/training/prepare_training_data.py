from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "distilbert-base-uncased"

DATA_FILES = {
    "train": "ml/datasets/processed/train.csv",
    "validation": "ml/datasets/processed/validation.csv",
    "test": "ml/datasets/processed/test.csv",
}


def prepare_training_data():

    print("Loading processed datasets...")

    dataset = load_dataset("csv", data_files=DATA_FILES)

    print("\nDatasets loaded:")
    print(dataset)

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=256
        )

    print("\nTokenizing datasets...")

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Rename labels to the standard name expected by Transformers
    tokenized_dataset = tokenized_dataset.rename_column("labels", "label")

    # Keep only the columns needed for training
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    print("\nTokenization complete!")
    print(tokenized_dataset)

    print("\nExample tokenized record:")
    print(tokenized_dataset["train"][0])


if __name__ == "__main__":
    prepare_training_data()
