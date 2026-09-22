from backend.services.model_service import predict_depression


def analyze_text(text: str):
    label, confidence = predict_depression(text)

    # In our current dataset/model convention:
    # label 1 = depression-related
    # label 0 = non-depression-related

    if label == 1:
        depression_probability = confidence

        if confidence >= 0.80:
            risk_level = "high"
        elif confidence >= 0.60:
            risk_level = "medium"
        else:
            risk_level = "low"

        emotion = "depression-related"
        message = (
            "The model detected language associated with depression-related content. "
            "This is an AI classification signal, not a medical diagnosis."
        )

    else:
        depression_probability = 1.0 - confidence

        if depression_probability >= 0.80:
            risk_level = "high"
        elif depression_probability >= 0.60:
            risk_level = "medium"
        else:
            risk_level = "low"

        emotion = "neutral"
        message = (
            "The model did not detect strong depression-related language. "
            "This is an AI classification signal, not a medical diagnosis."
        )

    return {
        "text": text,
        "depression_probability": depression_probability,
        "emotion": emotion,
        "risk_level": risk_level,
        "message": message,
    }
