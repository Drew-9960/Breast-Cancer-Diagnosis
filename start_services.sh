#!/bin/bash

echo "Starting ML pipeline services..."

# Activate virtual environment
source .venv/bin/activate

# Start MLflow Tracking Server
echo "Starting MLflow Tracking Server on port 5001..."
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5001 &

# Wait a few seconds to ensure MLflow is up
sleep 5

# Start FastAPI Model Inference Server
echo "Starting FastAPI Inference API on port 8000..."
uvicorn src.inference_api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit Dashboard
echo "Launching Streamlit Dashboard on port 8501..."
streamlit run src/streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0 &

echo "All services are now running!"
