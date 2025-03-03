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
mlflow.set_tracking_uri("http://127.0.0.1:5001")
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

        # Save model if it's the best so far
        global best_model, best_accuracy, best_params
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            joblib.dump(model, "model/best_model.pkl")

        return accuracy

# Initialize best model tracking
best_model = None
best_accuracy = 0
best_params = {}

# Run Optuna optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# ✅ Save training summary after best model is found
summary = f"""
Best Model Training Run:
- Accuracy: {best_accuracy:.4f}
- Parameters: {best_params}
"""
with open("model/training_summary.txt", "w") as f:
    f.write(summary)

print("\n✅ Training summary saved: model/training_summary.txt")
print("\n✅ Model Training Complete!")
