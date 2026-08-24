from fastapi import APIRouter

from backend.schemas.analysis import AnalysisRequest, AnalysisResponse

from backend.services.analyzer import analyze_text


router = APIRouter(prefix="/api", tags=["Analysis"])


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest):

    result = analyze_text(request.text)

    return result
