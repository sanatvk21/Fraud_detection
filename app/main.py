from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


# Create app
app = FastAPI()


# Load model
model = joblib.load(
    "models/xgb_model.pkl"
)


# Request schema
class Transaction(BaseModel):

    features: list[float]


# Home route
@app.get("/")
def home():

    return {
        "message":
        "Fraud Detection API Running"
    }


# Prediction route
@app.post("/predict")
def predict(data: Transaction):

    features = np.array(
        data.features
    ).reshape(1, -1)


    prediction = model.predict(
        features
    )

    probability = model.predict_proba(
        features
    )[0][1]


    return {

        "prediction":
        int(prediction[0]),

        "fraud_probability":
        float(probability)
    }