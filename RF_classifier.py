import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# === Load normalized data ===
df = pd.read_csv("features_windowed_s2_normalized.csv")

# === Bin KSS into 3 fatigue classes ===
def bin_kss(kss):
    if kss <= 3:
        return 0  # Low
    elif kss <= 6:
        return 1  # Medium
    else:
        return 2  # High

df["FatigueLevel"] = df["KSS"].apply(bin_kss)

# === Prepare features and labels ===
X = df[["PERCLOS", "BlinkRate", "YawnRate"]].values
y = df["FatigueLevel"].values

# === Train/Val/Test Split ===
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# === Scale features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# === Train Random Forest Classifier ===
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_train_scaled, y_train)

# === Predict and evaluate ===
y_pred = clf.predict(X_test_scaled)

# === Save and print classification report ===
report = classification_report(y_test, y_pred, target_names=["Low (1–3)", "Medium (4–6)", "High (7–9)"])
with open("rf_classification_report.txt", "w") as f:
    f.write(report)

print("=== Classification Report ===")
print(report)

# === Confusion Matrix Visualization ===
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low (1–3)", "Medium (4–6)", "High (7–9)"],
            yticklabels=["Low (1–3)", "Medium (4–6)", "High (7–9)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Random Forest)")
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png")
plt.show()
