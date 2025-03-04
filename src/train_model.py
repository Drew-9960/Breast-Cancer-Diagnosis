import mlflow
import optuna
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
X = pd.read_csv("data/processed_features.csv")
y = pd.read_csv("data/processed_labels.csv").values.ravel()

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configure MLflow for experiment tracking
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("Breast Cancer Tuning")

# Initialize variables to track the best model
best_model = None
best_accuracy = 0
best_params = {}

def objective(trial):
    """
    Objective function for Optuna hyperparameter tuning.

    - Defines search space for XGBoost hyperparameters.
    - Trains an XGBoost model using trial parameters.
    - Evaluates model performance on the test set.
    - Logs parameters and accuracy to MLflow.
    - Saves the best performing model.

    Returns:
        accuracy (float): Model accuracy on the test set.
    """
    with mlflow.start_run():
        # Define hyperparameter search space
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }

        # Train model
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Make predictions and evaluate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log experiment details to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        # Track best model
        global best_model, best_accuracy, best_params
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            joblib.dump(model, "model/best_model.pkl")

        return accuracy

# Run hyperparameter optimization with Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Save best model training summary
summary = f"""
Best Model Training Run:
- Accuracy: {best_accuracy:.4f}
- Parameters: {best_params}
"""
summary_path = "model/training_summary.txt"
with open(summary_path, "w") as f:
    f.write(summary)

print(f"\nTraining summary saved: {summary_path}")
print("\nModel Training Complete!")
