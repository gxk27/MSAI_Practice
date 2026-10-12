
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Reproducible demonstration data
rng = np.random.default_rng(42)

# Simulate 1,000 ordinary network connections
normal_traffic = pd.DataFrame({
    "duration": rng.exponential(2, 1000),
    "bytes_sent": rng.normal(1500, 300, 1000).clip(100),
    "bytes_received": rng.normal(5000, 1000, 1000).clip(100),
    "packets": rng.poisson(25, 1000).clip(1),
})

# Simulate 20 unusual connections for testing
test_anomalies = pd.DataFrame({
    "duration": rng.uniform(0.01, 0.2, 20),
    "bytes_sent": rng.uniform(30000, 100000, 20),
    "bytes_received": rng.uniform(0, 300, 20),
    "packets": rng.integers(200, 1000, 20),
})

# Train on ordinary traffic only
model = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42,
)

model.fit(normal_traffic)

# Combine ordinary and test traffic
test_data = pd.concat(
    [normal_traffic, test_anomalies],
    ignore_index=True,
)

# Isolation Forest: -1 = anomaly, 1 = normal
predictions = model.predict(test_data)

test_data["prediction"] = np.where(
    predictions == -1,
    "ANOMALY",
    "NORMAL",
)

# Mark which rows are simulated anomalies.
# This is test-only ground truth, not a model input.
test_data["is_simulated_anomaly"] = (
    np.arange(len(test_data)) >= len(normal_traffic)
)

# Save results for inspection
test_data.to_csv("detection_results.csv", index=False)

# Report basic evaluation metrics
actual = test_data["is_simulated_anomaly"].to_numpy()
flagged = predictions == -1

true_positives = np.sum(flagged & actual)
false_positives = np.sum(flagged & ~actual)
false_negatives = np.sum(~flagged & actual)

precision = true_positives / max(np.sum(flagged), 1)
recall = true_positives / max(np.sum(actual), 1)

print("\n=== Cyber Anomaly Detector ===")
print(f"Total connections: {len(test_data)}")
print(f"Connections flagged: {np.sum(flagged)}")
print(f"True positives: {true_positives}")
print(f"False positives: {false_positives}")
print(f"False negatives: {false_negatives}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")

print("\nSample flagged connections:")
print(
    test_data.loc[
        flagged,
        ["duration", "bytes_sent", "bytes_received",
         "packets", "prediction"]
    ].head(10).to_string(index=False)
)

print("\nResults saved to detection_results.csv")