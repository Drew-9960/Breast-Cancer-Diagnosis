import streamlit as st
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.tracking import MlflowClient

# Load dataset for visualization
df = pd.read_csv("data/processed_features.csv")

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

st.title("Breast Cancer Prediction & Monitoring Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio(
    "Select an option:", 
    ["Home", "Data Visualization", "Make Prediction", "Model Comparison", "Drift Monitoring"]
)

# Home Page
if options == "Home":
    st.write("## Welcome to the Breast Cancer Prediction Dashboard")
    st.markdown("""
    - **Train and deploy models automatically** using MLflow.
    - **Visualize feature distributions and detect data drift.**
    - **Compare different model versions and track performance.**
    - **Make real-time predictions using the latest trained model.**
    """)

# Data Visualization
elif options == "Data Visualization":
    st.subheader("Feature Distribution")

    # Select a feature for visualization
    feature = st.selectbox("Select a feature to visualize:", df.columns)

    # Plot distribution
    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax)
    st.pyplot(fig)

    # Show correlation heatmap if selected
    if st.checkbox("Show Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
        st.pyplot(fig)

# Make Prediction
elif options == "Make Prediction":
    st.subheader("Enter Patient Data for Prediction")

    # Create input fields for user data
    input_data = {feature: st.number_input(f"{feature}", value=0.0) for feature in df.columns}

    # Convert input to dataframe
    input_df = pd.DataFrame([input_data])

    # Send request to API for prediction
    if st.button("Predict"):
        try:
            response = requests.post("http://127.0.0.1:8000/predict/", json=input_data)
            prediction = response.json().get("prediction", "Error retrieving prediction")
            st.success(f"Prediction: {prediction}")
        except requests.exceptions.RequestException:
            st.error("Error: Unable to connect to the prediction API.")

# Model Comparison (MLflow)
elif options == "Model Comparison":
    st.subheader("Compare Model Versions")

    # Retrieve registered models
    models = client.search_model_versions("name='BreastCancerModel'")
    
    # Display model versions and metrics
    for model in models:
        st.write(f"**Version {model.version}** - {model.current_stage}")
        run = client.get_run(model.run_id)
        st.write(f"- Accuracy: {run.data.metrics.get('accuracy', 'N/A'):.4f}")
        st.write(f"- Registered: {model.creation_timestamp}")
        st.write("---")

    # Visualizing Optuna hyperparameter tuning results
    st.subheader("Hyperparameter Optimization Results")

    try:
        st.subheader("Optimization History")
        st.components.v1.html(open("reports/optuna_optimization_history.html").read(), height=600)

        st.subheader("Parallel Coordinates")
        st.components.v1.html(open("reports/optuna_parallel_coordinates.html").read(), height=600)

        st.subheader("Feature Importance")
        st.components.v1.html(open("reports/optuna_param_importances.html").read(), height=600)
    except FileNotFoundError:
        st.error("Error: Optuna visualization reports not found.")

# Drift Monitoring
elif options == "Drift Monitoring":
    st.subheader("Feature Drift Detection")

    try:
        drift_results = pd.read_csv("reports/drift_report.csv")
        st.write(drift_results)

        drifted_features = drift_results[drift_results["Drift Detected"] == "Yes"]
        if not drifted_features.empty:
            st.warning("Data drift detected in the following features:")
            st.write(drifted_features)
        else:
            st.success("No significant data drift detected.")

    except FileNotFoundError:
        st.error("Error: Drift report not found. Run `drift_detector.py` first.")
