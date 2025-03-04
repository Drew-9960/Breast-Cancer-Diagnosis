#!/bin/bash

echo "Stopping any running services..."

# Kill any existing MLflow, FastAPI, and Streamlit processes
pkill -f "mlflow server"
pkill -f "uvicorn src.inference_api:app"
pkill -f "streamlit run src/streamlit_dashboard.py"

echo "All previous services have been stopped."

echo "Starting the ML pipeline..."

# Verify Dataset
echo "Running dataset verification..."
python src/verify_dataset.py

# Run Feature Engineering
echo "Running feature engineering..."
python src/feature_engineering.py

# Detect Data Drift
echo "Running drift detection..."
python src/drift_detector.py

# Train Model
echo "Training the model..."
python src/train_model.py

# Start All Services (MLflow, API, Streamlit)
echo "Launching all services..."
bash deployment/start_services.sh

echo "ML pipeline is fully operational."


