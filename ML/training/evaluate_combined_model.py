from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# --------------------------------
# Paths
# --------------------------------

MODEL_DIR = Path("ml/models/combined_distilbert")
TEST_FILE = Path("ml/datasets/combined/test.csv")


# --------------------------------
# Load test data
# --------------------------------

print("Loading test dataset...")

test_df = pd.read_csv(TEST_FILE)

print(f"Test rows: {len(test_df)}")

test_df["labels"] = test_df["labels"].astype(int)

test_dataset = Dataset.from_pandas(
    test_df[["text", "labels"]],
    preserve_index=False,
)


# --------------------------------
# Load tokenizer
# --------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))


# --------------------------------
# Tokenize
# --------------------------------


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128,
    )


print("Tokenizing test dataset...")

test_dataset = test_dataset.map(
    tokenize,
    batched=True,
)

test_dataset = test_dataset.remove_columns(["text"])

test_dataset.set_format("torch")


# --------------------------------
# Load trained model
# --------------------------------

print("Loading trained combined model...")

model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))


# --------------------------------
# Metrics
# --------------------------------


def compute_metrics(pred):

    predictions = np.argmax(
        pred.predictions,
        axis=1,
    )

    labels = pred.label_ids

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# --------------------------------
# Evaluation
# --------------------------------

training_args = TrainingArguments(
    output_dir="ml/models/evaluation_temp",
    per_device_eval_batch_size=8,
    report_to="none",
)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


trainer = Trainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# --------------------------------
# Run test evaluation
# --------------------------------

print("\n================================")
print("EVALUATING COMBINED MODEL")
print("================================\n")

results = trainer.evaluate(test_dataset)


# --------------------------------
# Print results
# --------------------------------

print("\n================================")
print("FINAL TEST RESULTS")
print("================================")

print(f"Accuracy : {results['eval_accuracy']:.4f}")
print(f"Precision: {results['eval_precision']:.4f}")
print(f"Recall   : {results['eval_recall']:.4f}")
print(f"F1 Score : {results['eval_f1']:.4f}")

print("================================")
print("\nEvaluation complete. No training was performed.")
