#!/bin/bash

echo "Starting the ML pipeline..."

# Set environment variables for local execution
export MLFLOW_TRACKING_URI="http://localhost:5001"
export API_HOST="0.0.0.0"
export API_PORT=8000
export STREAMLIT_PORT=8501

# Step 1: Verify Dataset
echo "Running dataset verification..."
python src/verify_dataset.py

# Step 2: Run Feature Engineering
echo "Running feature engineering..."
python src/feature_engineering.py

# Step 3: Train Model
echo "Training the model..."
python src/train_model.py

# Step 4: Start API Server
echo "Starting API server..."
nohup python src/inference_api.py &

# Step 5: Start Streamlit Dashboard
echo "Launching Streamlit dashboard..."
nohup streamlit run src/streamlit_dashboard.py --server.port $STREAMLIT_PORT --server.address 0.0.0.0 &

echo "All services are now running."

