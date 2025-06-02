from pathlib import Path
import csv
from datetime import datetime
from collections import defaultdict

# === CONFIG ===
input_dir = Path("./window_pre-req")
output_csv = "features_windowed.csv"

fps = 6  # Adjusted for every-5th-frame extraction
window_size_sec = 10
stride_sec = 1
window_size = window_size_sec * fps
stride = stride_sec * fps

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6

# === Helper functions ===
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

# === Collect all data ===
print("🔍 Collecting frame-wise data...")
data = []
total_files = 0
skipped_files = 0

for txt_file in sorted(input_dir.glob("*_face0.txt")):
    total_files += 1
    base = txt_file.stem
    kss_path = input_dir / f"{base.split('_face0')[0]}.kss"
    time_path = input_dir / f"{base.split('_face0')[0]}.time"

    if not kss_path.exists() or not time_path.exists():
        skipped_files += 1
        continue

    ear, mar = load_features(txt_file)
    if ear is None or mar is None:
        skipped_files += 1
        continue

    timestamp = load_time_file(time_path)
    kss = load_kss(kss_path)

    if timestamp is None or kss is None:
        skipped_files += 1
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

print(f"✅ Loaded data from {total_files} files ({len(data)} valid, {skipped_files} skipped)\n")

# === Group by video ===
video_data = defaultdict(list)
for item in data:
    video_data[item["video"]].append(item)

# === Apply windowing and compute features ===
rows = []
print("🚀 Starting time window processing...")

for video_num, (video, frames) in enumerate(video_data.items(), 1):
    frames = sorted(frames, key=lambda x: x["timestamp"])
    total_windows = 0

    for i in range(0, len(frames) - window_size + 1, stride):
        window = frames[i:i+window_size]
        ears = [f["ear"] for f in window]
        mars = [f["mar"] for f in window]

        perclos = sum(1 for e in ears if e < EAR_THRESHOLD) / len(ears)
        blink_count = sum(1 for j in range(1, len(ears)) if ears[j-1] >= EAR_THRESHOLD and ears[j] < EAR_THRESHOLD)
        blink_rate = blink_count / window_size_sec if window_size_sec else 0
        yawn_count = sum(1 for m in mars if m > MAR_THRESHOLD)
        yawn_rate = yawn_count / window_size_sec if window_size_sec else 0
        # DEBUG: print all MAR values for one sample window
        if video == "4-3":  # target a known drowsy video
            print(f"\n📊 MARs for window starting at frame {window[0]['frame']}:")
            print(mars)

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
        total_windows += 1

    print(f"📦 [{video_num}/{len(video_data)}] Processed '{video}': {len(frames)} frames → {total_windows} windows")

print(f"\n🎯 Total windowed feature vectors: {len(rows)}")

# === Write output CSV ===
print(f"\n💾 Saving to '{output_csv}'...")
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Video", "Frame", "PERCLOS", "BlinkRate", "YawnRate", "YawnCount", "KSS", "Timestamp"])
    writer.writerows(rows)

print("✅ Done. Windowed features saved successfully.")
