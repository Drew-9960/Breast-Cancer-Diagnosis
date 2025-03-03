import mlflow
import optuna
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
X = pd.read_csv("data/processed_features.csv")
y = pd.read_csv("data/processed_labels.csv").values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Breast Cancer Tuning")

def objective(trial):
    with mlflow.start_run():
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        joblib.dump(model, "model/best_model.pkl")

        return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
print("\n✅ Model Training Complete!")