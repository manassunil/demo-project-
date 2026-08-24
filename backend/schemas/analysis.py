from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")


class AnalysisResponse(BaseModel):
    text: str
    depression_probability: float
    emotion: str
    risk_level: str
    message: str
