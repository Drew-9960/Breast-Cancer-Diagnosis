#!/bin/bash

echo "🚀 Starting the ML pipeline..."

# 1️⃣ Verify Dataset
echo "✅ Running dataset verification..."
python src/verify_dataset.py

# 2️⃣ Run Feature Engineering
echo "✅ Running feature engineering..."
python src/feature_engineering.py

# 3️⃣ Train Model
echo "✅ Training the model..."
python src/train_model.py

# 4️⃣ Start API Server
echo "✅ Starting API server..."
nohup python src/inference_api.py &

# 5️⃣ Start Streamlit Dashboard
echo "✅ Launching Streamlit dashboard..."
nohup streamlit run src/streamlit_dashboard.py &

echo "✅ Everything is up and running!"
