from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = Path("ML/models/depression_distilbert")


print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"Model loaded on: {device}")


def predict(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    predicted_label = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_label].item())

    return predicted_label, confidence


if __name__ == "__main__":
    text = input("\nEnter text: ").strip()

    if not text:
        print("No text entered.")
        raise SystemExit

    label, confidence = predict(text)

    print("\nPrediction:")
    print(f"Label: {label}")
    print(f"Model confidence: {confidence * 100:.2f}%")
