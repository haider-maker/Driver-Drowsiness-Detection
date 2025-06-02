import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
import matplotlib.pyplot as plt


# === Load Data ===
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("Y_train.csv").values.ravel()

X_val = pd.read_csv("X_val.csv")
y_val = pd.read_csv("Y_val.csv").values.ravel()

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("Y_test.csv").values.ravel()

# === Define Hyperparameter Grid ===
param_grid = {
    'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01],
    'activation': ['relu', 'tanh']
}

# === Grid Search Setup ===
mlp = MLPRegressor(max_iter=500, random_state=42)
grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='r2', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# === Retrieve Best Model ===
best_model = grid_search.best_estimator_
print("\n✅ Best Hyperparameters:", grid_search.best_params_)

# === Evaluate on Validation Set ===
val_preds = best_model.predict(X_val)
val_mse = mean_squared_error(y_val, val_preds)
val_r2 = r2_score(y_val, val_preds)
print(f"📊 Validation MSE: {val_mse:.4f}")
print(f"📈 Validation R² : {val_r2:.4f}")

# === Evaluate on Test Set ===
test_preds = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds)
test_r2 = r2_score(y_test, test_preds)
print(f"🧪 Test MSE: {test_mse:.4f}")
print(f"🧪 Test R² : {test_r2:.4f}")

# === Save Model ===
dump(best_model, "mlp_fatigue_model_tuned.pkl")
print("💾 Model saved to 'mlp_fatigue_model_tuned.pkl'")

# === Plot Predictions: Validation Set ===
plt.figure(figsize=(6, 5))
plt.scatter(y_val, val_preds, color='blue', alpha=0.6, edgecolor='k', label='Val')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--', label='Ideal')
plt.xlabel("True KSS (Validation)")
plt.ylabel("Predicted KSS")
plt.title("Predicted vs True KSS (Validation Set)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Plot Predictions: Test Set ===
plt.figure(figsize=(6, 5))
plt.scatter(y_test, test_preds, color='green', alpha=0.6, edgecolor='k', label='Test')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal')
plt.xlabel("True KSS (Test)")
plt.ylabel("Predicted KSS")
plt.title("Predicted vs True KSS (Test Set)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
