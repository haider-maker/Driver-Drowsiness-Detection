import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# === CONFIG ===
input_csv = "features_windowed_s2.csv"
output_csv = "features_windowed_s2_normalized.csv"
scaler_path = "scaler_s2.pkl"

# === Load data ===
df = pd.read_csv(input_csv)

# === Features to normalize ===
features = ["PERCLOS", "BlinkRate", "YawnRate"]
X = df[features]

# === Fit scaler and transform ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Save normalized data into DataFrame ===
df_scaled = df.copy()
df_scaled[features] = X_scaled

# === Save the normalized CSV ===
df_scaled.to_csv(output_csv, index=False)

# === Save the scaler for inference ===
joblib.dump(scaler, scaler_path)

print(f"✅ Normalization done. Saved to {output_csv}")
print(f"💾 Scaler saved as {scaler_path}")
