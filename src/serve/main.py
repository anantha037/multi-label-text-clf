from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from contextlib import asynccontextmanager
from src.serve.model import ModelLoader

# Global model instance
model_instance = ModelLoader()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    try:
        model_instance.load()
        print("Model successfully loaded on startup.")
    except Exception as e:
        print(f"Warning: Failed to load model on startup: {e}")
        # Not stopping the app entirely; allows /health to be hit before model exists
    yield
    # Shutdown logic

# Create FastAPI app
app = FastAPI(
    title="Multi-Label Text Classifier",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic Schemas
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text: str
    labels: List[str]
    scores: Dict[str, float]

@app.get("/health")
def health_check():
    """Returns the health status and whether the model is loaded."""
    return {
        "status": "ok",
        "model_loaded": model_instance.is_loaded
    }

@app.get("/labels")
def get_labels():
    """Returns the full list of possible label names."""
    if not model_instance.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded. Cannot retrieve labels.")
    return list(model_instance.mlb.classes_)

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Predicts the labels for the input text."""
    if not model_instance.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please ensure the model exists in the configured path.")
    
    try:
        result = model_instance.predict(request.text)
        return PredictResponse(
            text=request.text,
            labels=result["labels"],
            scores=result["scores"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.serve.main:app", host="0.0.0.0", port=8000)
