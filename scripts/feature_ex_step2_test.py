from pathlib import Path
import csv
from datetime import datetime
from collections import defaultdict

# === CONFIG ===
input_dir = Path("./window_pre-req")
output_csv = "features_windowed_smoothed.csv"

fps = 6  # Adjusted for every-5th-frame extraction
window_size_sec = 15
stride_sec = 1
window_size = window_size_sec * fps
stride = stride_sec * fps

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.3
SMOOTHING_KERNEL = 3

# === Helper Functions ===
def load_time_file(path):
    try:
        with path.open("r") as f:
            line = f.readline().strip()
            parts = line.split(":")
            if len(parts) != 4:
                return None
            hour, minute, second, millisecond = parts
            dt = datetime.strptime(f"{hour}:{minute}:{second}", "%H:%M:%S")
            microsecond = int(millisecond) * 1000
            dt = dt.replace(microsecond=microsecond)
            return dt
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
            line = f.read().strip()
            ear, mar = map(float, line.split())
            return ear, mar
    except:
        return None, None

def smooth_moving_avg(values, kernel_size=3):
    if len(values) < kernel_size:
        return values
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - kernel_size // 2)
        end = min(len(values), i + kernel_size // 2 + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed

# === Collect All Data ===
print("🔍 Collecting frame-wise data...")
data = []
for txt_file in sorted(input_dir.glob("*_face0.txt")):
    base = txt_file.stem
    kss_path = input_dir / f"{base.split('_face0')[0]}.kss"
    time_path = input_dir / f"{base.split('_face0')[0]}.time"

    if not kss_path.exists() or not time_path.exists():
        continue

    ear, mar = load_features(txt_file)
    timestamp = load_time_file(time_path)
    kss = load_kss(kss_path)

    if None in [ear, mar, timestamp, kss]:
        continue

    video_prefix = base.split("_")[0]
    frame_name = base + ".jpg"

    data.append({
        "video": video_prefix,
        "frame": frame_name,
        "ear": ear,
        "mar": mar,
        "timestamp": timestamp,
        "kss": kss
    })

# === Group by Video ===
video_data = defaultdict(list)
for item in data:
    video_data[item["video"]].append(item)

# === Windowing and Feature Extraction ===
rows = []
print("🚀 Starting time window processing...")
for video_num, (video, frames) in enumerate(video_data.items(), 1):
    frames = sorted(frames, key=lambda x: x["timestamp"])

    for i in range(0, len(frames) - window_size + 1, stride):
        window = frames[i:i+window_size]
        ears = [f["ear"] for f in window]
        mars = [f["mar"] for f in window]

        smoothed_mars = smooth_moving_avg(mars, kernel_size=SMOOTHING_KERNEL)

        perclos = sum(1 for e in ears if e < EAR_THRESHOLD) / len(ears)
        blink_count = sum(1 for j in range(1, len(ears)) if ears[j-1] >= EAR_THRESHOLD and ears[j] < EAR_THRESHOLD)
        blink_rate = blink_count / window_size_sec if window_size_sec else 0
        yawn_count = sum(1 for m in smoothed_mars if m > MAR_THRESHOLD)
        yawn_rate = yawn_count / window_size_sec if window_size_sec else 0

        mid_frame = window[len(window) // 2]
        rows.append([
            video,
            mid_frame["frame"],
            f"{perclos:.4f}",
            f"{blink_rate:.4f}",
            f"{yawn_rate:.4f}",
            yawn_count,
            mid_frame["kss"],
            mid_frame["timestamp"]
        ])

    print(f"📦 [{video_num}/{len(video_data)}] Processed '{video}': {len(frames)} frames")

# === Write to CSV ===
print(f"\n💾 Saving to '{output_csv}'...")
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Video", "Frame", "PERCLOS", "BlinkRate", "YawnRate", "YawnCount", "KSS", "Timestamp"])
    writer.writerows(rows)

print("✅ Done. Smoothed features saved successfully.")
