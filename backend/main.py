from fastapi import FastAPI

app = FastAPI(
    title="MindLens AI",
    description="Real-time AI-based mental health language risk analysis",
    version="0.1.0",
)


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
