from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

from backend.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "project": "MindLens AI",
        "status": "running",
        "message": "Backend is working!",
    }


def test_analyze_response_fields():
    expected_result = {
        "text": "sample text",
        "depression_probability": 0.9,
        "emotion": "depression-related",
        "risk_level": "high",
        "message": "model result",
    }

    with patch("backend.api.analyze.analyze_text", return_value=expected_result):
        response = client.post("/api/analyze", json={"text": "sample text"})

    assert response.status_code == 200
    assert set(response.json()) == {
        "text",
        "depression_probability",
        "emotion",
        "risk_level",
        "message",
    }
    assert response.json() == expected_result


@patch("backend.api.analyze.analyze_text")
def test_analyze_accepts_valid_text(mock_analyze):
    mock_analyze.return_value = {
        "text": "sample text",
        "depression_probability": 0.5,
        "emotion": "neutral",
        "risk_level": "low",
        "message": "model result",
    }

    response = client.post("/api/analyze", json={"text": "sample text"})

    assert response.status_code == 200
    mock_analyze.assert_called_once_with("sample text")


@patch("backend.api.analyze.analyze_text")
def test_analyze_rejects_empty_text(mock_analyze):
    response = client.post("/api/analyze", json={"text": ""})

    assert response.status_code == 422
    mock_analyze.assert_not_called()


@patch("backend.api.analyze.analyze_text")
def test_analyze_rejects_text_over_5000_characters(mock_analyze):
    response = client.post("/api/analyze", json={"text": "x" * 5001})

    assert response.status_code == 422
    mock_analyze.assert_not_called()


def test_risk_levels_are_calculated_by_analyzer():
    from backend.services.analyzer import analyze_text

    with patch("backend.services.analyzer.predict_depression", return_value=(1, 0.9)):
        high_result = analyze_text("high risk")
    with patch("backend.services.analyzer.predict_depression", return_value=(1, 0.7)):
        medium_result = analyze_text("medium risk")
    with patch("backend.services.analyzer.predict_depression", return_value=(0, 0.9)):
        low_result = analyze_text("low risk")

    assert high_result["risk_level"] == "high"
    assert medium_result["risk_level"] == "medium"
    assert low_result["risk_level"] == "low"
    assert low_result["depression_probability"] == pytest.approx(0.1)
