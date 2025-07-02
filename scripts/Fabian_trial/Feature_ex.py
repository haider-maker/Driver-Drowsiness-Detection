from pathlib import Path
import csv

# === CONFIG ===
video_id = "Fabian_KSS_8_Vid_1"
input_dir = Path("data/Fabian_trail_EAR_MAR_output")
output_csv = "Fabian_features_windowed_yawn_distinct.csv"

fps = 20
window_size_sec = 30
stride_sec = 10
window_size = window_size_sec * fps
stride = stride_sec * fps

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.42

# === Helper function ===
def load_features(path):
    try:
        with path.open("r") as f:
            line = f.read().strip()
            ear, mar = map(float, line.split())
            return ear, mar
    except Exception as e:
        print(f"[⛔] Failed to read EAR/MAR from {path}: {e}")
        return None, None

# === Extract KSS from video name ===
try:
    parts = video_id.split("_")
    kss_value = int(parts[2])
except Exception as e:
    print(f"[⛔] Failed to extract KSS from video_id {video_id}: {e}")
    kss_value = None

print(f"🎬 Video ID: {video_id}")
print(f"🎯 KSS Value: {kss_value}")

# === Collect all frame data ===
data = []
skipped_files = 0

txt_files = sorted(input_dir.glob("frame_*.txt"))
print(f"📂 Found {len(txt_files)} frame files in {input_dir}")

for i, txt_file in enumerate(txt_files, 1):
    frame_id = txt_file.stem

    ear, mar = load_features(txt_file)
    if ear is None or mar is None:
        print(f"[🚫] Skipping invalid or missing data: {txt_file.name}")
        skipped_files += 1
        continue

    data.append({
        "video": video_id,
        "frame": frame_id + ".jpg",
        "ear": ear,
        "mar": mar,
        "kss": kss_value
    })

    if i % 100 == 0:
        print(f"   ✔️ Processed {i} frames so far...")

print(f"\n✅ Total valid frames collected: {len(data)}")
print(f"⛔ Skipped frames due to errors: {skipped_files}")

# === Sort by frame number ===
data_sorted = sorted(data, key=lambda x: int(x["frame"].split("_")[1].split(".")[0]))

# === Apply windowing ===
rows = []
total_windows = 0

for i in range(0, len(data_sorted) - window_size + 1, stride):
    window = data_sorted[i:i + window_size]
    ears = [d["ear"] for d in window]
    mars = [d["mar"] for d in window]

    if len(ears) != window_size:
        print(f"[⚠️] Incomplete window at index {i}")
        continue

    perclos = sum(1 for e in ears if e < EAR_THRESHOLD) / len(ears)

    # Blink detection
    blink_count = 0
    blinking = False
    last_blink_end = -999
    MIN_BLINK_GAP = 6

    for idx, ear in enumerate(ears):
        if ear < EAR_THRESHOLD and not blinking and idx - last_blink_end > MIN_BLINK_GAP:
            blink_count += 1
            blinking = True
        elif ear >= EAR_THRESHOLD and blinking:
            blinking = False
            last_blink_end = idx

    blink_rate = blink_count / window_size_sec

    # Yawn detection
    yawn_count = 0
    yawning = False
    last_yawn_end = -999
    MIN_GAP = 30

    for idx, mar in enumerate(mars):
        if mar > MAR_THRESHOLD and not yawning and idx - last_yawn_end > MIN_GAP:
            yawn_count += 1
            yawning = True
        elif mar <= MAR_THRESHOLD and yawning:
            yawning = False
            last_yawn_end = idx

    yawn_rate = yawn_count / window_size_sec

    mid_frame = window[len(window) // 2]
    rows.append([
        video_id,
        mid_frame["frame"],
        f"{perclos:.4f}",
        f"{blink_rate:.4f}",
        f"{yawn_rate:.4f}",
        yawn_count,
        mid_frame["kss"]
    ])

    total_windows += 1
    if total_windows % 10 == 0:
        print(f"   🧮 Windows processed: {total_windows}")

print(f"\n🎯 Total windows extracted: {total_windows}")

# === Save CSV ===
print(f"💾 Saving CSV to {output_csv}")
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Video",
        "Frame",
        "PERCLOS",
        "BlinkRate",
        "YawnRate",
        "YawnCount",
        "KSS"
    ])
    writer.writerows(rows)

print("✅ Done!")
