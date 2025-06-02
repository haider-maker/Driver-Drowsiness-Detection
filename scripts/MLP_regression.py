import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# === Load Data ===
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("Y_train.csv").values.ravel()
X_val = pd.read_csv("X_val.csv")
y_val = pd.read_csv("Y_val.csv").values.ravel()
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("Y_test.csv").values.ravel()

# === Define MLP Model ===
# model = MLPRegressor(
#     hidden_layer_sizes=(64, 32),
#     activation='logistic',  # ← sigmoid activation
#     max_iter=500,
#     random_state=42
# )

model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='tanh', max_iter=500, random_state=42)


# === Train ===
model.fit(X_train, y_train)

# === Predict ===
val_preds = model.predict(X_val)
test_preds = model.predict(X_test)

# === Clip predictions to [0, 9] ===
val_preds = np.clip(val_preds, 0, 9)
test_preds = np.clip(test_preds, 0, 9)

# === Evaluate ===
val_mse = mean_squared_error(y_val, val_preds)
val_r2 = r2_score(y_val, val_preds)
test_mse = mean_squared_error(y_test, test_preds)
test_r2 = r2_score(y_test, test_preds)

print(f"📊 Validation MSE: {val_mse:.4f}")
print(f"📈 Validation R² : {val_r2:.4f}")
print(f"🧪 Test MSE: {test_mse:.4f}")
print(f"🧪 Test R² : {test_r2:.4f}")

# === Save Model ===
joblib.dump(model, "mlp_fatigue_model_clipped.pkl")
print("✅ Model saved to mlp_fatigue_model_clipped.pkl")
