from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import joblib

app = FastAPI()

pipeline = joblib.load(
    'churn_pipeline.pkl'
)


class ClientData(BaseModel):
    Tenure: float
    Support_Calls: float
    Payment_Delay: float


@app.get("/")
def root():
    return {
        "message": "Churn Prediction API"
    }


@app.post("/predict")
def predict(data: ClientData):

    df = pd.DataFrame([{
        'Tenure': data.Tenure,
        'Support Calls': data.Support_Calls,
        'Payment Delay': data.Payment_Delay
    }])

    prediction = pipeline.predict(df)[0]

    probability = pipeline.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability)
    }