from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "distilbert/distilbert-base-uncased"


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

print("Model loaded successfully!")
print("Model:", MODEL_NAME)
