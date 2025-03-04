import pandas as pd
from scipy.stats import ks_2samp

def check_data_drift():
    """
    Detects potential data drift by comparing distributions of historical and new data
    using the Kolmogorov-Smirnov test.

    - Randomly samples data from `processed_features.csv` for comparison.
    - Computes the KS test for each feature.
    - Flags drift if the p-value is below 0.05.
    - Saves the results to `reports/drift_report.csv`.

    Output:
        - A CSV report indicating whether drift was detected for each feature.
    """

    # Load historical and new data samples
    old_data = pd.read_csv("data/processed_features.csv").sample(50, random_state=42)
    new_data = pd.read_csv("data/processed_features.csv").sample(50, random_state=99)

    # Perform drift detection for each feature
    drift_results = []
    for feature in old_data.columns:
        p_value = ks_2samp(old_data[feature], new_data[feature]).pvalue
        drift_results.append({
            "Feature": feature,
            "P-Value": round(p_value, 4),  # Round for readability
            "Drift Detected": "Yes" if p_value < 0.05 else "No"
        })

    # Save results to a CSV file
    df_drift = pd.DataFrame(drift_results)
    df_drift.to_csv("reports/drift_report.csv", index=False)

    print("Drift detection complete. Results saved to 'reports/drift_report.csv'.")

if __name__ == "__main__":
    check_data_drift()

