# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(title="Churn Prediction API", version="1.0")

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = "model/churn_model.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully")
    else:
        print("❌ Model file not found!")

# Input schema
class CustomerData(BaseModel):
    tenure: float           # months as customer
    monthly_charges: float  # monthly bill
    total_charges: float    # total amount paid
    has_internet: int       # 1 = yes, 0 = no
    has_phone: int          # 1 = yes, 0 = no

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running!", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: CustomerData):
    if model is None:
        return {"error": "Model not loaded"}
    
    features = np.array([[
        data.tenure,
        data.monthly_charges,
        data.total_charges,
        data.has_internet,
        data.has_phone
    ]])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    return {
        "churn_prediction": int(prediction),
        "churn_label": "Will Churn" if prediction == 1 else "Will NOT Churn",
        "churn_probability": round(float(probability[1]), 4),
        "retain_probability": round(float(probability[0]), 4)
    }