import uvicorn
import mlflow.pyfunc
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import os

# Load environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://breast_cancer_mlflow:5001")

# Initialize FastAPI application
app = FastAPI(title="Breast Cancer Prediction API")

# Define the input data schema
class PredictionInput(BaseModel):
    radius1: float
    texture1: float
    perimeter1: float
    area1: float
    smoothness1: float
    compactness1: float
    concavity1: float
    concave_points1: float
    symmetry1: float
    fractal_dimension1: float
    radius2: float
    texture2: float
    perimeter2: float
    area2: float
    smoothness2: float
    compactness2: float
    concavity2: float
    concave_points2: float
    symmetry2: float
    fractal_dimension2: float
    radius3: float
    texture3: float
    perimeter3: float
    area3: float
    smoothness3: float
    compactness3: float
    concavity3: float
    concave_points3: float
    symmetry3: float
    fractal_dimension3: float

# Try loading the model from MLflow first
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.pyfunc.load_model("models:/BreastCancerModel/Production")
    model_source = "MLflow"
except Exception as e:
    print(f"Warning: Failed to load MLflow model: {e}. Loading local model instead.")
    model = joblib.load("model/best_model.pkl")
    model_source = "Local joblib model"

@app.get("/")
def home():
    return {"message": f"Breast Cancer Prediction API is running. Model source: {model_source}"}

@app.post("/predict/")
def predict(data: PredictionInput):
    input_data = pd.DataFrame([data.dict()])
    prediction = model.predict(input_data)
    diagnosis = "Malignant" if prediction[0] == 1 else "Benign"
    return {"prediction": diagnosis}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
