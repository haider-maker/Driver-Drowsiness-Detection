import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load files
df_feat = pd.read_csv("Feature_vector.csv")
df_pred = pd.read_csv("real_captured_fatigue_classified_30_70.csv")

# Merge on Timestamp (ensure both are string for merge)
df_feat["Timestamp"] = df_feat["Timestamp"].astype(str)
df_pred["timestamp"] = df_pred["timestamp"].astype(str)
df = pd.merge(df_feat, df_pred, left_on="Timestamp", right_on="timestamp", suffixes=("_feat", "_pred"))

print(f"Merged rows: {len(df)} (out of {len(df_feat)} features, {len(df_pred)} classified)")

# --- Handle column names for fatigue classes ---
fatigue_col_map = {
    "CF": "fatigue_camera_level" if "fatigue_camera_level" in df.columns else "CF",
    "SF": "fatigue_steering_level" if "fatigue_steering_level" in df.columns else "SF",
    "LF": "fatigue_lane_level" if "fatigue_lane_level" in df.columns else "LF"
}

# --- Crosstab and count/box/violin plots ---
for col, colname in fatigue_col_map.items():
    print(f"\n=== {col} vs KSS ===")
    print(pd.crosstab(df[colname], df["KSS"], margins=True))
    plt.figure(figsize=(8,4))
    sns.countplot(x="KSS", hue=colname, data=df, palette="Set2")
    plt.title(f"{col} class vs KSS (count)")
    plt.show()
    plt.figure(figsize=(8,4))
    sns.boxplot(x=colname, y="KSS", data=df, palette="Set2")
    plt.title(f"KSS distribution for each {col} class")
    plt.show()

# --- Correlation heatmap ---
fatigue_map = {"Low": 1, "Moderate": 2, "High": 3}
df_corr = df.copy()
for col, colname in fatigue_col_map.items():
    df_corr[col + "_num"] = df_corr[colname].map(fatigue_map)
corr_cols = ["KSS", "CF_num", "SF_num", "LF_num"]
corr = df_corr[corr_cols].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation heatmap: KSS vs Fatigue Scores")
plt.show()

# --- Grouped bar chart for all three classifiers ---
kss_labels = sorted(df["KSS"].unique())
cf_ct = pd.crosstab(df[fatigue_col_map["CF"]], df["KSS"]).reindex(["Low", "Moderate", "High"])
sf_ct = pd.crosstab(df[fatigue_col_map["SF"]], df["KSS"]).reindex(["Low", "Moderate", "High"])
lf_ct = pd.crosstab(df[fatigue_col_map["LF"]], df["KSS"]).reindex(["Low", "Moderate", "High"])

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fatigue_levels = ["Low", "Moderate", "High"]
colors = ['green', 'orange', 'red']
bar_width = 0.25
x = np.arange(len(kss_labels))

def plot_grouped(ax, ct, title):
    for i, level in enumerate(fatigue_levels):
        ax.bar(x + (i - 1)*bar_width, ct.loc[level, kss_labels], 
               width=bar_width, label=level, color=colors[i])
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(kss_labels)
    ax.set_xlabel("KSS Score")
    ax.set_ylabel("Count")
    ax.legend(title="Fatigue Class")

plot_grouped(axes[0], cf_ct, "CF vs KSS")
plot_grouped(axes[1], sf_ct, "SF vs KSS")
plot_grouped(axes[2], lf_ct, "LF vs KSS")

plt.suptitle("Grouped Bar Chart: CF / SF / LF Class Distribution Across KSS", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Violin plots for KSS distribution per fatigue class ---
plt.figure(figsize=(18, 5))
for i, (col, colname) in enumerate(fatigue_col_map.items(), 1):
    plt.subplot(1, 3, i)
    sns.violinplot(x=colname, y="KSS", data=df, palette="Set2", inner="box")
    plt.title(f"KSS distribution for {col}")
    plt.xlabel(col)
    plt.ylabel("KSS" if i == 1 else "")
plt.tight_layout()
plt.show()

# --- Crosstab bar/histogram plots for each classifier output ---
for col, colname in fatigue_col_map.items():
    ct = pd.crosstab(df[colname], df["KSS"])
    # Bar chart (grouped)
    ct.T.plot(kind="bar", figsize=(8,5))
    plt.title(f"{col} class distribution across KSS (grouped bar)")
    plt.xlabel("KSS")
    plt.ylabel("Count")
    plt.legend(title=col)
    plt.tight_layout()
    plt.show()
    # Bar chart (stacked)
    ct.T.plot(kind="bar", stacked=True, figsize=(8,5))
    plt.title(f"{col} class distribution across KSS (stacked bar)")
    plt.xlabel("KSS")
    plt.ylabel("Count")
    plt.legend(title=col)
    plt.tight_layout()
    plt.show()
    # Histogram for KSS per class
    plt.figure(figsize=(8,5))
    for fatigue_class in ["Low", "Moderate", "High"]:
        kss_vals = df[df[colname] == fatigue_class]["KSS"]
        plt.hist(kss_vals, bins=range(1,11), alpha=0.5, label=fatigue_class)
    plt.title(f"KSS histogram for each {col} class")
    plt.xlabel("KSS")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()