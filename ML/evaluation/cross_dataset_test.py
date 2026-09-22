from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

REDDIT_TEST = Path("ML/datasets/processed/test.csv")
MHC_TEST = Path("ML/datasets/mental_health_corpus/processed/test.csv")

REDDIT_MODEL = Path("ML/models/depression_distilbert")
MHC_MODEL = Path("ML/models/mhc_distilbert")


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# --------------------------------------------------
# Load model
# --------------------------------------------------


def load_model(model_path):
    print(f"\nLoading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.to(device)
    model.eval()

    return tokenizer, model


# --------------------------------------------------
# Predict
# --------------------------------------------------


def predict_dataset(
    model,
    tokenizer,
    df,
    batch_size=32,
):
    predictions = []

    texts = df["text"].tolist()

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        batch_predictions = (
            torch.argmax(
                outputs.logits,
                dim=-1,
            )
            .cpu()
            .tolist()
        )

        predictions.extend(batch_predictions)

        completed = min(start + batch_size, len(texts))

        print(f"\rProcessed {completed}/{len(texts)}", end="")

    print()

    return predictions


# --------------------------------------------------
# Evaluation
# --------------------------------------------------


def evaluate_model(
    model_name,
    model_path,
    dataset_name,
    dataset_path,
):
    print("\n" + "=" * 70)
    print(f"MODEL: {model_name} | DATASET: {dataset_name}")
    print("=" * 70)

    df = pd.read_csv(dataset_path)

    print(f"Samples: {len(df)}")

    tokenizer, model = load_model(model_path)

    predictions = predict_dataset(
        model,
        tokenizer,
        df,
    )

    labels = df["labels"].astype(int).tolist()

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    matrix = confusion_matrix(
        labels,
        predictions,
    )

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    print("\nConfusion matrix:")
    print(matrix)

    return {
        "model": model_name,
        "dataset": dataset_name,
        "samples": len(df),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------


def main():

    reddit_results = pd.read_csv(REDDIT_TEST)

    mhc_results = pd.read_csv(MHC_TEST)

    print("\nTest dataset sizes:")
    print(f"Reddit: {len(reddit_results)}")
    print(f"MHC:    {len(mhc_results)}")

    results = []

    # ----------------------------------------------
    # Reddit model -> Reddit test
    # ----------------------------------------------

    results.append(
        evaluate_model(
            "Reddit DistilBERT",
            REDDIT_MODEL,
            "Reddit",
            REDDIT_TEST,
        )
    )

    # ----------------------------------------------
    # Reddit model -> MHC test
    # ----------------------------------------------

    results.append(
        evaluate_model(
            "Reddit DistilBERT",
            REDDIT_MODEL,
            "MHC",
            MHC_TEST,
        )
    )

    # ----------------------------------------------
    # MHC model -> MHC test
    # ----------------------------------------------

    results.append(
        evaluate_model(
            "MHC DistilBERT",
            MHC_MODEL,
            "MHC",
            MHC_TEST,
        )
    )

    # ----------------------------------------------
    # MHC model -> Reddit test
    # ----------------------------------------------

    results.append(
        evaluate_model(
            "MHC DistilBERT",
            MHC_MODEL,
            "Reddit",
            REDDIT_TEST,
        )
    )

    # ----------------------------------------------
    # Summary
    # ----------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n\n" + "=" * 70)
    print("CROSS-DATASET SUMMARY")
    print("=" * 70)

    print(
        results_df[
            [
                "model",
                "dataset",
                "samples",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(index=False)
    )

    output_dir = Path("ML/evaluation")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_dir / "cross_dataset_results.csv",
        index=False,
    )

    print(f"\nResults saved to: {output_dir / 'cross_dataset_results.csv'}")


if __name__ == "__main__":
    main()
