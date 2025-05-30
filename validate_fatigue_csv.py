import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === CONFIG ===
csv_file = "features_windowed_s2.csv"  # Change if needed
video_to_plot = "6-2"               # Pick one to visualize first
save_plots = True                   # Save the plots to disk

# === LOAD DATA ===
df = pd.read_csv(csv_file)

# === BASIC VALIDATION ===
print("✅ Columns:", list(df.columns))
print("🔍 Total rows:", len(df))
print("📊 Unique videos:", df["Video"].nunique())
print("🧪 KSS range:", df["KSS"].min(), "to", df["KSS"].max())

# === VALUE CHECKS ===
print("\nChecking value ranges...")
print("Max PERCLOS:", df["PERCLOS"].max())
print("Max Blink Rate:", df["BlinkRate"].max())
print("Max Yawn Rate:", df["YawnRate"].max())

assert (df["PERCLOS"] <= 1.0).all(), "❌ PERCLOS > 1.0 found"
assert (df["PERCLOS"] >= 0.0).all(), "❌ PERCLOS < 0.0 found"
assert (df["BlinkRate"] >= 0.0).all(), "❌ Negative BlinkRate found"
assert (df["YawnRate"] >= 0.0).all(), "❌ Negative YawnRate found"

# === GROUP BY VIDEO ===
videos = df["Video"].unique()

# === VISUALIZATION ===
for video in videos:
    df_video = df[df["Video"] == video].copy()
    df_video["Timestamp"] = pd.to_datetime(df_video["Timestamp"], format="%H:%M:%S:%f")

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(x="Timestamp", y="PERCLOS", data=df_video, label="PERCLOS", marker='o')
    sns.lineplot(x="Timestamp", y="BlinkRate", data=df_video, label="Blink Rate", marker='x')
    sns.lineplot(x="Timestamp", y="YawnRate", data=df_video, label="Yawn Rate", marker='s')

    # KSS coloring (optional)
    for i, row in df_video.iterrows():
        kss = row["KSS"]
        ax.annotate(f"{kss}", (row["Timestamp"], row["PERCLOS"]), fontsize=7, alpha=0.6, color='gray')

    plt.title(f"Fatigue Feature Trends - Video: {video}")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_plots:
        Path("validation_plots").mkdir(exist_ok=True)
        plt.savefig(f"validation_plots/{video}_features_plot.png")

    if video == video_to_plot:
        plt.show()
    else:
        plt.close()

print("\n✅ All plots generated. Check 'validation_plots/' folder if save_plots=True.")
