from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.analyze import router as analyze_router


app = FastAPI(
    title="MindLens AI",
    description="Real-time AI-based mental health language risk analysis",
    version="0.1.0",
)


# Allow requests from our browser extension/frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(analyze_router)


@app.get("/")
def root():
    return {
        "project": "MindLens AI",
        "status": "running",
        "message": "Backend is working!",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}
