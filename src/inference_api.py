from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Load latest "Production" model from MLflow
model = mlflow.pyfunc.load_model("models:/BreastCancerModel/Production")

app = FastAPI()

@app.post("/predict/")
def predict(features: dict):
    df = pd.DataFrame([features])
    prediction = model.predict(df)
    return {"prediction": "Benign" if prediction[0] == 0 else "Malignant"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
