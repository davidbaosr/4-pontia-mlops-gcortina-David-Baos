from pathlib import Path
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "encoders.pkl"

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)


class PredictionInput(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


@app.get("/")
def root():
    return {"message": "API running"}


@app.post("/predict")
def predict(data: PredictionInput):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)
    decoded = label_encoder.inverse_transform(pred)
    return {"prediction": decoded[0]}