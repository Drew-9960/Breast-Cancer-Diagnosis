import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    """
    Loads the breast cancer dataset, preprocesses it, and saves the processed features and labels.

    - Drops the 'ID' column if it exists.
    - Converts the 'Diagnosis' column ('M' -> 1, 'B' -> 0).
    - Normalizes all feature values using StandardScaler.
    - Saves the processed feature set and labels as separate CSV files.
    """

    # Load dataset
    df = pd.read_csv("data/breast_cancer_data.csv")

    # Remove 'ID' column if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Convert 'Diagnosis' column to numeric (Malignant = 1, Benign = 0)
    df["Diagnosis"] = df["Diagnosis"].replace({"M": 1, "B": 0})

    # Separate features (X) and target (y)
    X = df.drop(columns=["Diagnosis"])
    y = df["Diagnosis"]

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # Save processed data
    X.to_csv("data/processed_features.csv", index=False)
    y.to_csv("data/processed_labels.csv", index=False)

    print("\nData preprocessing complete. Processed datasets saved.")

if __name__ == "__main__":
    load_and_preprocess_data()
