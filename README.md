# Breast Cancer Diagnosis Model

> **Note:** ⚠️ In Development, docker deployment is functional within a virtual env.  
{: .highlight-yellow}

## Overview
This project is a **machine learning pipeline** for classifying **breast cancer tumors** as **benign** or **malignant** using the **Breast Cancer Wisconsin (Diagnostic) Dataset**. The pipeline includes data preprocessing, feature engineering, model training, hyperparameter tuning, model versioning with MLflow, and deployment using **FastAPI** and **Streamlit**.

## Features
- **Data Processing**: Cleans and preprocesses raw data.
- **Model Training**: Uses **XGBoost** with **Optuna** for hyperparameter tuning.
- **Experiment Tracking**: Logs model metrics and versions in **MLflow**.
- **Inference API**: FastAPI server for making predictions.
- **Streamlit Dashboard**: Visualizes model performance and predictions.
- **Docker Support**: Can be deployed in a containerized environment.
- **Kubernetes Deployment**: Supports scaling and orchestration.

---

## Repository Structure
```
Breast-Cancer-Diagnosis/
│
├── data/                      # Stores processed datasets
│   ├── breast_cancer_data.csv  # Original dataset
│   ├── processed_features.csv # Cleaned dataset
│   ├── processed_labels.csv # Cleaned dataset
│
├── model/                     # Stores trained models
│   ├── best_model.pkl         # Best trained model
│   ├── training_summary.txt   # Model metadata
│
├── reports/                     # Stores reports
│   ├── drift_reports.csv         # drift results
│
├── src/                       # Source code
│   ├── feature_engineering.py   # Data preprocessing
│   ├── train_model.py           # ML training & tuning
│   ├── inference_api.py         # FastAPI for inference
│   ├── streamlit_dashboard.py   # Dashboard for monitoring
│   ├── verify_dataset.py        # Data validation
│   ├── drift_detector.py        # Detecting data drift
│   ├── _init_.py                # init
│   ├── config.py                # Config
│
├── Kubernetes/                 # Kubernetes manifests
│   ├── deployment.yaml        
│   ├── service.yaml               
│
├── Dockerfile                # Dockerfile
├── docker-compose.yml        # Docker setup
├── run.sh                   # Full ML pipeline execution
├── start_services.sh         # Starts API and dashboard
├── mlflow.db                   # MLflow experiment tracking database
├── mlruns/                      # MLflow model artifacts
├── requirements.txt             # Python dependencies
├── README.md                    # Documentation
```

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Drew-9960/Breast-Cancer-Diagnosis.git
cd Breast-Cancer-Diagnosis
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start MLflow Tracking Server
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5001 &
```

---

## Running the Pipeline
### 1. Verify Dataset
```bash
python src/verify_dataset.py
```
### 2. Run Feature Engineering
```bash
python src/feature_engineering.py
```
### 3. Train Model
```bash
python src/train_model.py
```
### 4. Start API Server
```bash
python src/inference_api.py
```
### 5. Start Streamlit Dashboard
```bash
streamlit run src/streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0
```

Alternatively, run everything in one command:
```bash
bash deployment/run.sh
```

---

## Using the Model
### **Making Predictions via API**
Send a POST request with sample input:
```bash
curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: application/json" -d '{"radius1": 17.99, "texture1": 10.38, ...}'
```

### **Accessing MLflow Experiment Tracking**
Go to:
```bash
http://127.0.0.1:5001
```

### **Viewing the Streamlit Dashboard**
Open in browser:
```bash
http://127.0.0.1:8501
```

---

## Deployment Options
### **1. Docker Deployment**
Build and run using Docker Compose:
```bash
docker-compose up --build
```

---

## Future Improvements
- Implement real-time model retraining with new data.
- Automate drift detection and model replacement.
- Enhance API security with authentication.
- Deploy on cloud-based Kubernetes clusters.

---

## Acknowledgments
This project is based on the **Breast Cancer Wisconsin (Diagnostic) Dataset**, provided by the **UCI Machine Learning Repository**.

