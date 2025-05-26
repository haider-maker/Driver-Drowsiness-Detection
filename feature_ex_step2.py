from pathlib import Path
import csv
from datetime import datetime
from collections import defaultdict

# === CONFIG ===
features_dir = Path("./features_output")
timestamps_dir = Path("./cropped_faces")
output_csv = "features_windowed.csv"

fps = 30
window_size_sec = 30
stride_sec = 10
window_size = window_size_sec * fps
stride = stride_sec * fps

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.75

# === Helper functions ===
def load_time_file(path):
    try:
        with path.open("r") as f:
            t = f.readline().strip().split()
            return datetime.strptime(" ".join(t[3:]), "%H %M %S %f")
    except:
        return None

def load_kss(path):
    try:
        with path.open("r") as f:
            return int(f.read().strip())
    except:
        return None

def load_features(path):
    try:
        with path.open("r") as f:
            line = f.read().strip()
            ear, mar = map(float, line.split())
            return ear, mar
    except:
        return None, None

# === Collect all data first ===
data = []

for feat_path in sorted(features_dir.glob("*.txt")):
    base = feat_path.stem  # e.g. 1-1_frame_0000_face0
    parts = base.split("_")
    if len(parts) < 2:
        continue

    video = parts[0]
    frame = "_".join(parts[1:]) + ".jpg"

    time_base = base.split("_face")[0]  # strips "_face0"
    time_path = timestamps_dir / f"{time_base}.time"
    kss_path = timestamps_dir / f"{time_base}.kss"

    print(f"✅ Checking: {feat_path.name} → {time_path.name}, {kss_path.name}")


    if not time_path.exists() or not kss_path.exists():
        continue

    ear, mar = load_features(feat_path)
    if ear is None or mar is None:
        continue

    timestamp = load_time_file(time_path)
    kss = load_kss(kss_path)

    if timestamp is None or kss is None:
        continue

    data.append({
        "video": video,
        "frame": frame,
        "ear": ear,
        "mar": mar,
        "timestamp": timestamp,
        "kss": kss
    })

print(f"✅ Loaded {len(data)} valid frames.")

# === Group by video ===
video_data = defaultdict(list)
for d in data:
    video_data[d["video"]].append(d)

# === Sliding Window and feature extraction ===
rows = []
for video, frames in video_data.items():
    frames = sorted(frames, key=lambda x: x["timestamp"])
    for i in range(0, len(frames) - window_size + 1, stride):
        window = frames[i:i+window_size]
        ears = [f["ear"] for f in window]
        mars = [f["mar"] for f in window]

        perclos = sum(1 for e in ears if e < EAR_THRESHOLD) / len(ears)
        blink_count = sum(
            1 for j in range(1, len(ears))
            if ears[j-1] >= EAR_THRESHOLD and ears[j] < EAR_THRESHOLD
        )
        blink_rate = blink_count / window_size_sec
        yawn_count = sum(1 for m in mars if m > MAR_THRESHOLD)
        yawn_rate = yawn_count / window_size_sec

        mid_frame = window[len(window) // 2]
        rows.append([
            video,
            mid_frame["frame"],
            f"{perclos:.4f}",
            f"{blink_rate:.4f}",
            f"{yawn_rate:.4f}",
            mid_frame["kss"],
            mid_frame["timestamp"]
        ])

print(f"📊 Extracted {len(rows)} windowed feature vectors.")

# === Write CSV ===
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Video", "Frame", "PERCLOS", "BlinkRate", "YawnRate", "KSS", "Timestamp"])
    writer.writerows(rows)

print(f"✅ Saved: {output_csv}")
