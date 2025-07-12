from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict
from datetime import datetime
import uuid
import model  # your existing model.py
import joblib
import pandas as pd
import tensorflow as tf  # Import TensorFlow here
import asyncio


app = FastAPI()

# Load or train model at startup
ml_model = model.load_or_train_model()
SCALER_PATH = "/tmp/scaler.pkl"


# -------------------------
# Request & Response Schemas
# -------------------------

class FamilyInput(BaseModel):
    adult_male: int
    adult_female: int
    child: int

class UserInput(BaseModel):
    user_id: str
    region: str
    season: str
    event: str
    family: FamilyInput
    stock: Dict[str, float]  # product_name: quantity

class RetrainRequest(BaseModel):
    user_id: str


# -------------------------
# API Routes
# -------------------------

# Define request body schema
class Item(BaseModel):
    name: str
    quantity: int



@app.get("/")
def read_root():
    return {"message": "✅ GrocyGenie API is running."}



@app.post("/testpost")
def test_post(item: Item):
    return {"message": f"Received item '{item.name}' with quantity {item.quantity}"}


@app.post("/predict")
def predict(input_data: UserInput):
    try:
        user_dict = input_data.dict()
        user_id = user_dict["user_id"]

        predictions = model.predict_user_input(user_dict)

        model.store_predictions(user_id, predictions, user_dict)

        feedback = pd.DataFrame([{
            'date': datetime.today().strftime('%Y-%m-%d'),
            'product': k,
            'region': user_dict['region'],
            'season': user_dict['season'],
            'event': user_dict['event'],
            'adult_male': user_dict['family']['adult_male'],
            'adult_female': user_dict['family']['adult_female'],
            'child': user_dict['family']['child'],
            'consumption': v['predicted_consumption'],
            'finish_error': v['predicted_finish_error'],
            'finish_days': v['predicted_finish_days']
        } for k, v in predictions.items()])

        model.insert_feedback(user_id, feedback)

        return {
            "user_id": user_id,
            "predictions": predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain")
def retrain_model(request: RetrainRequest):
    success = model.retrain_model_with_feedback(request.user_id)
    if success:
        # Reload model and scaler globally for future predictions
        model.ml_model = tf.keras.models.load_model(model.MODEL_PATH)
        model.scaler = joblib.load(SCALER_PATH)
        return {"message": f"Model retrained using feedback for user {request.user_id}."}
    else:
        raise HTTPException(status_code=404, detail="No feedback found for retraining.")
