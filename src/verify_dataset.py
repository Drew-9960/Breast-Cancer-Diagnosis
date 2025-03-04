import pandas as pd

# Load dataset manually from UCI repository
url = "https://archive.ics.uci.edu/static/public/17/data.csv"
df = pd.read_csv(url)

# Display dataset shape
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# Display column names for verification
print("\nDataset Columns:")
print(df.columns.tolist())

# Ensure required columns exist
if "ID" in df.columns and "Diagnosis" in df.columns:
    print("\n'ID' and 'Diagnosis' columns successfully loaded.")
else:
    print("\n'ID' or 'Diagnosis' column is missing. Check dataset structure.")

# Save the dataset locally for consistency
dataset_path = "data/breast_cancer_data.csv"
df.to_csv(dataset_path, index=False)
print(f"\nDataset saved locally at '{dataset_path}'.")

# Extract features (X) and target variable (y)
X = df.drop(columns=["ID", "Diagnosis"])  # Remove ID and target column
y = df["Diagnosis"]

# Convert 'Diagnosis' column ('M' for Malignant, 'B' for Benign) to numeric values
y = y.replace({"M": 1, "B": 0})

# Display retrieved feature columns
print("\nFeature Columns Retrieved:")
print(X.columns.tolist())

# Verify class distribution
print("\nClass Distribution:")
print(y.value_counts())

# Display first few rows of features and target labels
print("\nFirst 5 Rows of Features:")
print(X.head())

print("\nFirst 5 Labels:")
print(y.head())

print("\nData verification completed.")
