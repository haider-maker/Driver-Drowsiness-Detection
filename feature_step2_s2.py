from pathlib import Path
import csv
from collections import defaultdict

# === CONFIGURATION ===
input_dir = Path("./window_pre-req_s2")
output_csv = "features_windowed_s2.csv"

fps = 6  # Extracted every 5th frame from 30 FPS original
window_size_sec = 60
stride_sec = 10
window_size = window_size_sec * fps  # 360 frames
stride = stride_sec * fps            # 60 frames

EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.42

# === Helper functions ===
def load_time_file(path):
    try:
        with path.open("r") as f:
            parts = f.readline().strip().split(":")
            if len(parts) != 4:
                return None
            return ":".join(parts)
    except Exception:
        return None

def load_kss(path):
    try:
        with path.open("r") as f:
            val = f.read().strip()
            return int(val) if val else None
    except:
        return None

def load_features(path):
    try:
        with path.open("r") as f:
            ear, mar = map(float, f.read().strip().split())
            return ear, mar
    except:
        return None, None

# === Load all frame-wise data ===
print("📥 Loading frame-wise features and metadata...")
data = []
missing_files = 0
bad_entries = 0
total_files = 0

for txt_file in sorted(input_dir.glob("*.txt")):
    if txt_file.name.endswith(".kss") or txt_file.name.endswith(".time"):
        continue

    total_files += 1
    base = txt_file.stem
    kss_path = input_dir / f"{base}.kss"
    time_path = input_dir / f"{base}.time"

    if not kss_path.exists() or not time_path.exists():
        missing_files += 1
        continue

    ear, mar = load_features(txt_file)
    timestamp = load_time_file(time_path)
    kss = load_kss(kss_path)

    if None in (ear, mar, timestamp, kss):
        bad_entries += 1
        continue

    video = base.split("_")[0]  # e.g., "1-3"
    data.append({
        "video": video,
        "frame": base + ".jpg",
        "ear": ear,
        "mar": mar,
        "timestamp": timestamp,
        "kss": kss
    })

print(f"✅ Loaded: {len(data)} frames | Skipped (missing): {missing_files} | Bad entries: {bad_entries}")

# === Group by video ===
video_data = defaultdict(list)
for item in data:
    video_data[item["video"]].append(item)

print(f"🎬 Found {len(video_data)} unique videos")

# === Windowing and feature vector calculation ===
rows = []
print("📊 Calculating features for each window...")

for vid_idx, (video, frames) in enumerate(video_data.items(), 1):
    frames = sorted(frames, key=lambda x: x["timestamp"])
    window_count = 0

    for i in range(0, len(frames) - window_size + 1, stride):
        window = frames[i:i+window_size]
        ears = [f["ear"] for f in window]
        mars = [f["mar"] for f in window]

        perclos = sum(e < EAR_THRESHOLD for e in ears) / len(ears)
        blink_count = sum(1 for j in range(1, len(ears)) if ears[j-1] >= EAR_THRESHOLD and ears[j] < EAR_THRESHOLD)
        blink_rate = blink_count / (window_size_sec / 60)  # blinks/minute
        yawn_count = sum(1 for m in mars if m > MAR_THRESHOLD)
        yawn_rate = yawn_count / 5  # normalized to 5-minute pseudo-window

        center = window[len(window)//2]
        rows.append([
            video,
            center["frame"],
            f"{perclos:.4f}",
            f"{blink_rate:.4f}",
            f"{yawn_rate:.4f}",
            center["kss"],
            center["timestamp"]
        ])
        window_count += 1

    print(f"✅ [{vid_idx}/{len(video_data)}] {video} → {window_count} windows processed")

# === Save output to CSV ===
print(f"\n💾 Saving {len(rows)} feature vectors to {output_csv}...")
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Video", "Frame", "PERCLOS", "BlinkRate", "YawnRate", "KSS", "Timestamp"])
    writer.writerows(rows)

print("✅ Done.")
