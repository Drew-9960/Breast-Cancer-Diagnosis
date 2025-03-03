#!/bin/bash

echo "🚀 Starting ML Pipeline..."

# Step 1: Preprocess Data
echo "🔄 Preprocessing Data..."
python src/feature_engineering.py

# Step 2: Run Hyperparameter Tuning with Model Versioning
echo "⚙️ Running Hyperparameter Tuning & Auto-Deployment..."
python src/train_model.py

# Step 3: Check Data Drift
echo "📊 Running Drift Detection..."
python src/drift_detector.py

# Step 4: Start MLflow Server
echo "📈 Launching MLflow Tracking UI..."
nohup mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 &

# Step 5: Start FastAPI for Inference (loads latest Production model)
echo "🖥️ Starting API Server..."
nohup python src/inference_api.py &

# Step 6: Start Streamlit Dashboard with Optuna Plots
echo "📊 Launching Streamlit Dashboard..."
streamlit run src/streamlit_dashboard.py
