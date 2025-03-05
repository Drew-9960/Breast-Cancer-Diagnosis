import streamlit as st
import pandas as pd
import requests
import mlflow
from mlflow.tracking import MlflowClient
import os

# Load environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://breast_cancer_mlflow:5001")
API_URL = os.getenv("API_URL", "http://breast_cancer_api:8000")

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

st.title("Breast Cancer Prediction Dashboard")

# Introduction
st.write("### Welcome to the Breast Cancer Prediction Dashboard")
st.write(
    "This dashboard allows you to make real-time predictions for breast cancer "
    "and monitor model performance using MLflow."
)

# Prediction Section
st.subheader("Make a Prediction")

# Load feature names (you might need to adjust based on your dataset)
feature_names = [
    "radius1", "texture1", "perimeter1", "area1", "smoothness1",
    "compactness1", "concavity1", "concave_points1", "symmetry1", "fractal_dimension1",
    "radius2", "texture2", "perimeter2", "area2", "smoothness2",
    "compactness2", "concavity2", "concave_points2", "symmetry2", "fractal_dimension2",
    "radius3", "texture3", "perimeter3", "area3", "smoothness3",
    "compactness3", "concavity3", "concave_points3", "symmetry3", "fractal_dimension3"
]

# Create input fields
input_data = {feature: st.number_input(feature, value=0.0) for feature in feature_names}

# Convert input to dataframe
input_df = pd.DataFrame([input_data])

# Make Prediction Button
if st.button("Predict"):
    try:
        response = requests.post(f"{API_URL}/predict/", json=input_data)
        prediction = response.json().get("prediction", "Error retrieving prediction")
        st.success(f"Prediction: {prediction}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the prediction API. Ensure the API is running.\n\nError: {e}")

# Drift Monitoring Section
st.subheader("Data Drift Monitoring")
try:
    drift_results = pd.read_csv("reports/drift_report.csv")
    st.write(drift_results)

    drifted_features = drift_results[drift_results["Drift Detected"] == "Yes"]
    if not drifted_features.empty:
        st.warning("⚠️ Data drift detected in the following features:")
        st.write(drifted_features)
    else:
        st.success("No significant data drift detected.")
except FileNotFoundError:
    st.error("Error: Drift report not found.")
