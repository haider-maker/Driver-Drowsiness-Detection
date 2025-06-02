import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# === Load data ===
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()

# === Load model and scaler ===
model = joblib.load("mlp_fatigue_model.pkl")
scaler = joblib.load("scaler_s2.pkl")

# === Normalize test data ===
X_test_scaled = scaler.transform(X_test)

# === Predict ===
test_preds = model.predict(X_test_scaled)

# === Evaluate (optional) ===
mse = mean_squared_error(y_test, test_preds)
r2 = r2_score(y_test, test_preds)
print(f"📊 Test MSE: {mse:.4f}")
print(f"📈 Test R² : {r2:.4f}")

# === Plot ===
plt.figure(figsize=(8, 5))
plt.scatter(y_test, test_preds, alpha=0.6, color='blue', edgecolors='k')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel("True KSS")
plt.ylabel("Predicted KSS")
plt.title("MLP Prediction vs Ground Truth (Test Set)")
plt.grid(True)
plt.tight_layout()
plt.show()
