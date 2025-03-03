import pandas as pd
import pandas as pd

# Load dataset manually from UCI
url = "https://archive.ics.uci.edu/static/public/17/data.csv"
df = pd.read_csv(url)

# Print column names for verification
print("df.columns.count()")
print(df.rows.count())
print("\n✅ Dataset Columns:")
print(df.columns.tolist())

# Ensure ID and Diagnosis exist
if "ID" in df.columns and "Diagnosis" in df.columns:
    print("\n✅ 'ID' and 'Diagnosis' columns successfully loaded!")
else:
    print("\n❌ 'ID' or 'Diagnosis' column is missing!")

# Save the dataset locally for consistency
df.to_csv("data/breast_cancer_data.csv", index=False)
print("\n✅ Dataset saved locally at 'data/breast_cancer_data.csv'.")

# Extract Features (X) and Target (y)
X = df.drop(columns=['ID', 'Diagnosis'])  # Remove ID and target column
y = df['Diagnosis']  # Target variable

# Convert 'Diagnosis' (M/B) to numeric values (1 = Malignant, 0 = Benign)
y = y.replace({'M': 1, 'B': 0})

# Print column names for verification
print("\n✅ Feature Columns Retrieved from Dataset:")
print(X.columns.tolist())

# Verify class distribution
print("\n✅ Class Distribution:")
print(y.value_counts())

# Print first few rows
print("\n✅ First 5 Rows of Features:")
print(X.head())

# Print first few labels
print("\n✅ First 5 Labels:")
print(y.head())

print("\n✅ Data Verification Completed!")
