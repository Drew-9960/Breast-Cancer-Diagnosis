import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    df = pd.read_csv("data/breast_cancer_data.csv")

    # Drop 'ID' column
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Convert 'Diagnosis' to 1/0
    df["Diagnosis"] = df["Diagnosis"].replace({"M": 1, "B": 0})

    # Extract features and target
    X = df.drop(columns=["Diagnosis"])
    y = df["Diagnosis"]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # Save processed data
    X.to_csv("data/processed_features.csv", index=False)
    y.to_csv("data/processed_labels.csv", index=False)

    print("\n✅ Data Preprocessing Complete!")

if __name__ == "__main__":
    load_and_preprocess_data()