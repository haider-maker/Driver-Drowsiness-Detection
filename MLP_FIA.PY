import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load trained MLP model
model_path = "mlp_fatigue_model_tuned.pkl"
mlp_model = joblib.load(model_path)

# Extract input layer weights (first hidden layer)
input_weights = mlp_model.coefs_[0]  # shape: (num_features, hidden_units)

# Compute mean absolute weights per feature
feature_importance = np.mean(np.abs(input_weights), axis=1)

# Feature names in order: PERCLOS, BlinkRate, YawnRate
feature_names = ["PERCLOS", "BlinkRate", "YawnRate"]

# Plot feature importance
plt.figure(figsize=(6, 4))
plt.bar(feature_names, feature_importance, color='skyblue', edgecolor='black')
plt.title("Feature Importance from Input Layer Weights")
plt.ylabel("Mean Absolute Weight")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

feature_importance, feature_names
