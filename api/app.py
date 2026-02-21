from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from src.inference import predict_from_payload, load_model

app = FastAPI()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"

model = None


def _get_model():
    global model
    if model is None:
        try:
            model = load_model(MODEL_PATH)
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model not found at {MODEL_PATH}. "
                    "Train first using: python src\\train.py"
                ),
            )
    return model


@app.get("/")
def home():
    return {"message": "Ride Cancellation API"}


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class PredictionResponse(BaseModel):
    is_cancelled: int


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    try:
        prediction = predict_from_payload(data.model_dump(), _get_model())
        return PredictionResponse(is_cancelled=int(prediction))
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")
