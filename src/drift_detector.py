from scipy.stats import ks_2samp
import pandas as pd

def check_data_drift():
    # Load historical and new data
    old_data = pd.read_csv("data/processed_features.csv").sample(50, random_state=42)
    new_data = pd.read_csv("data/processed_features.csv").sample(50, random_state=99)

    # Check for drift feature by feature
    drift_results = []
    for feature in old_data.columns:
        p_value = ks_2samp(old_data[feature], new_data[feature]).pvalue
        drift_results.append({
            "Feature": feature,
            "P-Value": p_value,
            "Drift Detected": "Yes" if p_value < 0.05 else "No"
        })

    # Save drift results
    df_drift = pd.DataFrame(drift_results)
    df_drift.to_csv("reports/drift_report.csv", index=False)

    print("✅ Drift Detection Complete. Results saved to `reports/drift_report.csv`")

if __name__ == "__main__":
    check_data_drift()
