def analyze_text(text: str):
    """
    Temporary analysis function.

    This will be replaced with the real
    Transformer model in Step 4.
    """

    text_lower = text.lower()

    risk_words = [
        "hopeless",
        "worthless",
        "empty",
        "alone",
        "depressed",
        "sad",
        "tired of life",
    ]

    matches = [word for word in risk_words if word in text_lower]

    if len(matches) >= 2:
        probability = 0.85
        risk = "high"
        emotion = "sadness"

    elif len(matches) == 1:
        probability = 0.55
        risk = "medium"
        emotion = "sadness"

    else:
        probability = 0.10
        risk = "low"
        emotion = "neutral"

    return {
        "text": text,
        "depression_probability": probability,
        "emotion": emotion,
        "risk_level": risk,
        "message": "Temporary rule-based analysis",
    }
