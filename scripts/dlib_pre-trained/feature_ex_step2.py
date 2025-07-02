from pathlib import Path
import csv
from datetime import datetime
from collections import defaultdict

# === CONFIG ===
input_dir = Path("data/window_pre-req_s2")   # Now this is the main input folder
output_csv = "features_windowed_dlib.csv"

fps = 6
window_size_sec = 120  # 2 minutes
stride_sec = 10        # 1 minute
window_size = window_size_sec * fps  # 1800 frames
stride = stride_sec * fps            # 360 frames

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.42

# === Helper functions ===
def load_time_file(path):
    try:
        with path.open("r") as f:
            line = f.readline().strip()
            parts = line.split(":")
            if len(parts) != 4:
                print(f"[⛔] Invalid time format in {path}")
                return None
            hour, minute, second, millisecond = parts
            dt = datetime.strptime(f"{hour}:{minute}:{second}", "%H:%M:%S")
            microsecond = int(millisecond) * 1000
            return dt.replace(microsecond=microsecond)
    except Exception as e:
        print(f"[⛔] Failed to read timestamp from {path}: {e}")
        return None

def load_kss(path):
    try:
        with path.open("r") as f:
            val = f.read().strip()
            return int(val) if val else None
    except Exception as e:
        print(f"[⛔] Failed to read KSS from {path}: {e}")
        return None

def load_features(path):
    try:
        with path.open("r") as f:
            line = f.read().strip()
            ear, mar = map(float, line.split())
            return ear, mar
    except Exception as e:
        print(f"[⛔] Failed to read EAR/MAR from {path}: {e}")
        return None, None

# === Collect all data ===
print("🔍 Collecting frame-wise data...")
data = []
skipped_files = 0

all_folders = sorted(input_dir.glob("*"))
print(f"📁 Found {len(all_folders)} video folders in input_dir")

for video_folder in all_folders:
    if not video_folder.is_dir():
        print(f"⚠️  Skipping non-directory: {video_folder}")
        continue
    video_id = video_folder.name
    print(f"\n📂 Processing folder: {video_id}")

    frame_txts = sorted(video_folder.glob("frame_*.txt"))
    print(f"   📄 Found {len(frame_txts)} frame text files")

    for i, txt_file in enumerate(frame_txts, 1):
        frame_id = txt_file.stem  # e.g., frame_1234

        kss_path = video_folder / f"{frame_id}.kss"
        time_path = video_folder / f"{frame_id}.time"

        if not kss_path.exists():
            print(f"[🚫] Missing .kss file: {kss_path}")
            skipped_files += 1
            continue
        if not time_path.exists():
            print(f"[🚫] Missing .time file: {time_path}")
            skipped_files += 1
            continue

        ear, mar = load_features(txt_file)
        if ear is None or mar is None:
            print(f"[🚫] Skipping due to invalid features: {txt_file}")
            skipped_files += 1
            continue

        timestamp = load_time_file(time_path)
        if timestamp is None:
            print(f"[🚫] Invalid timestamp: {time_path}")
            skipped_files += 1
            continue

        kss = load_kss(kss_path)
        if kss is None:
            print(f"[🚫] Invalid KSS: {kss_path}")
            skipped_files += 1
            continue

        data.append({
            "video": video_id,
            "frame": frame_id + ".jpg",
            "ear": ear,
            "mar": mar,
            "timestamp": timestamp,
            "kss": kss
        })

        if i % 100 == 0:
            print(f"   ✔️ Processed {i} files so far in {video_id}...")

print(f"\n✅ Total valid records collected: {len(data)}")
print(f"⛔ Skipped files due to errors: {skipped_files}\n")

# === Group by video ===
video_data = defaultdict(list)
for item in data:
    video_data[item["video"]].append(item)

print(f"📊 Grouped data into {len(video_data)} video(s)")

# === Apply windowing ===
rows = []
print("🚀 Processing time windows...\n")

for idx, (video, frames) in enumerate(video_data.items(), 1):
    print(f"📹 [{idx}/{len(video_data)}] Video: {video}, Frames: {len(frames)}")
    frames = sorted(frames, key=lambda x: x["timestamp"])
    total_windows = 0

    for i in range(0, len(frames) - window_size + 1, stride):
        window = frames[i:i + window_size]
        ears = [f["ear"] for f in window]
        mars = [f["mar"] for f in window]

        if len(ears) != window_size:
            print(f"[⚠️] Incomplete window at index {i} for video {video}")
            continue

        perclos = sum(1 for e in ears if e < EAR_THRESHOLD) / len(ears)
        blink_count = sum(1 for j in range(1, len(ears)) if ears[j - 1] >= EAR_THRESHOLD and ears[j] < EAR_THRESHOLD)
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
            yawn_count,
            mid_frame["kss"],
            mid_frame["timestamp"]
        ])
        total_windows += 1

        if total_windows % 10 == 0:
            print(f"   🧮 Windows processed so far for {video}: {total_windows}")

    print(f"✅ Finished {video}: {total_windows} windows extracted\n")

print(f"\n🎯 Total windowed feature vectors: {len(rows)}")

# === Save CSV ===
print(f"💾 Saving output to '{output_csv}'...")
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Video", "Frame", "PERCLOS", "BlinkRate", "YawnRate", "YawnCount", "KSS", "Timestamp"])
    writer.writerows(rows)

print("✅ Done. Features saved successfully.")
