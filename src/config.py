import os

# Set MLflow Tracking URI based on environment (local or Docker)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Streamlit Dashboard Port
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))
