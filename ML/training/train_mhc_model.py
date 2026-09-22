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

DATA_DIR = Path("ML/datasets/mental_health_corpus/processed")
OUTPUT_DIR = Path("ML/models/mhc_distilbert")

TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

MODEL_NAME = "distilbert-base-uncased"


# --------------------------------
# Load datasets
# --------------------------------

print("Loading MHC datasets...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train rows: {len(train_df)}")
print(f"Validation rows: {len(val_df)}")
print(f"Test rows: {len(test_df)}")

train_df["labels"] = train_df["labels"].astype(int)
val_df["labels"] = val_df["labels"].astype(int)
test_df["labels"] = test_df["labels"].astype(int)


# --------------------------------
# Convert pandas -> HF Dataset
# --------------------------------

train_dataset = Dataset.from_pandas(
    train_df[["text", "labels"]],
    preserve_index=False,
)

val_dataset = Dataset.from_pandas(
    val_df[["text", "labels"]],
    preserve_index=False,
)

test_dataset = Dataset.from_pandas(
    test_df[["text", "labels"]],
    preserve_index=False,
)


# --------------------------------
# Tokenizer
# --------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128,
    )


print("Tokenizing datasets...")

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)


train_dataset = train_dataset.remove_columns(["text"])
val_dataset = val_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")


# --------------------------------
# Model
# --------------------------------

print("Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)


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
# Training settings
# --------------------------------

use_cuda = torch.cuda.is_available()

print(f"CUDA available: {use_cuda}")

if use_cuda:
    print(f"GPU: {torch.cuda.get_device_name(0)}")


training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    eval_strategy="epoch",
    # Don't keep multiple huge checkpoint copies.
    save_strategy="epoch",
    save_total_limit=1,
    logging_strategy="steps",
    logging_steps=100,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=use_cuda,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# --------------------------------
# Trainer
# --------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# --------------------------------
# Training
# --------------------------------

print("\nStarting MHC training...")

trainer.train()


# --------------------------------
# Validation
# --------------------------------

print("\nValidation results:")

validation_results = trainer.evaluate()

print(validation_results)


# --------------------------------
# Test
# --------------------------------

print("\nTest results:")

test_results = trainer.evaluate(test_dataset)

print(test_results)


# --------------------------------
# Save model
# --------------------------------

print("\nSaving MHC model...")

trainer.save_model(str(OUTPUT_DIR))

tokenizer.save_pretrained(str(OUTPUT_DIR))

print(f"\nMHC model saved to: {OUTPUT_DIR}")
