import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# === Load classified results ===
df = pd.read_csv("rule_based_predictions.csv")

y_true = df["ActualClass"]
y_pred = df["PredictedClass"]

# === Compute Metrics ===
print("=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=["Low (0–3)", "Medium (4–6)", "High (7–9)"]))
print("\n✅ Accuracy:", accuracy_score(y_true, y_pred))

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low (0–3)", "Medium (4–6)", "High (7–9)"],
            yticklabels=["Low (0–3)", "Medium (4–6)", "High (7–9)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Rule-Based Classifier")
plt.tight_layout()
plt.savefig("rule_based_confusion_matrix.png")
plt.show()
