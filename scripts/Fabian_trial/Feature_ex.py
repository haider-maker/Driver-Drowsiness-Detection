import csv
from pathlib import Path

# === CONFIGURATION ===
ear_mar_root = Path("data/Subjects_EAR_MAR_output")
sync_root = Path("data/Sync_Extracted_Data")
output_csv_all = Path("features_all_subjects_windowed.csv")
output_dir_subjects = Path("subject_csvs")
output_dir_subjects.mkdir(parents=True, exist_ok=True)

fps = 10
window_size_sec = 60
stride_sec = 20
window_size = window_size_sec * fps
stride = stride_sec * fps

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.46

# === Helper functions ===
def load_ear_mar(path):
    try:
        with open(path, "r") as f:
            line = f.read().strip()
            ear, mar = map(float, line.split())
            return ear, mar
    except Exception as e:
        print(f"[⛔] Failed to read EAR/MAR from {path}: {e}")
        return None, None

def load_timestamps(sync_csv_path):
    timestamps = {}
    if not sync_csv_path.exists():
        print(f"⚠️ No sync CSV found: {sync_csv_path}")
        return timestamps
    with open(sync_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ir_filename = row["ir_filename"]
            stem = Path(ir_filename).stem
            timestamps[stem] = row["timestamp"]
    return timestamps

# === Process all subjects ===
rows_all = []
rows_by_subject = {}

for subject_folder in sorted(ear_mar_root.glob("*")):
    if not subject_folder.is_dir():
        continue

    subject = subject_folder.name
    rows_subject = []

    print(f"\n👤 Processing subject: {subject}")

    for video_folder in sorted(subject_folder.glob("*")):
        if not video_folder.is_dir():
            continue

        video_id = video_folder.name

        # Extract KSS from video_id
        try:
            parts = video_id.split("_")
            kss_value = int(parts[1])
        except Exception as e:
            print(f"[⛔] Failed to extract KSS from {video_id}: {e}")
            kss_value = None

        # Load timestamp map
        sync_csv_path = (
            sync_root
            / subject
            / video_id
            / f"{video_id}_Sync.csv"
        )
        timestamp_map = load_timestamps(sync_csv_path)

        # Collect frame data
        frame_files = sorted(video_folder.glob("*.txt"))
        data = []
        for file in frame_files:
            frame_stem = file.stem

            ear, mar = load_ear_mar(file)
            if ear is None or mar is None:
                continue

            timestamp = timestamp_map.get(frame_stem, "")
            data.append({
                "video": video_id,
                "frame": f"{frame_stem}.png",
                "ear": ear,
                "mar": mar,
                "timestamp": timestamp,
                "kss": kss_value
            })

        if not data:
            print(f"⚠️ No valid frames for {video_id}")
            continue

        # Sort by frame timestamp float
        try:
            data_sorted = sorted(
                data,
                key=lambda x: float(Path(x["frame"]).stem)
            )
        except:
            data_sorted = data

        # Apply windowing
        total_windows = 0
        for i in range(0, len(data_sorted) - window_size + 1, stride):
            window = data_sorted[i : i + window_size]
            ears = [d["ear"] for d in window]
            mars = [d["mar"] for d in window]

            if len(ears) != window_size:
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

            mid_frame = window[len(window)//2]
            row = [
                video_id,
                mid_frame["frame"],
                f"{perclos:.4f}",
                f"{blink_rate:.4f}",
                f"{yawn_rate:.4f}",
                yawn_count,
                mid_frame["kss"],
                mid_frame["timestamp"],
            ]
            rows_subject.append(row)
            rows_all.append([subject] + row)

            total_windows += 1
            if total_windows % 10 == 0:
                print(f"   🧮 Windows for {video_id}: {total_windows}")

        print(f"✅ Finished {video_id}. Windows extracted: {total_windows}")

    # Save subject CSV
    if rows_subject:
        subject_csv = output_dir_subjects / f"{subject}_features_windowed.csv"
        with open(subject_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Video",
                "Frame",
                "PERCLOS",
                "BlinkRate",
                "YawnRate",
                "YawnCount",
                "KSS",
                "Timestamp",
            ])
            writer.writerows(rows_subject)
        print(f"✅ Saved CSV for subject {subject}: {subject_csv}")

# === Save combined CSV ===
if rows_all:
    with open(output_csv_all, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Subject",
            "Video",
            "Frame",
            "PERCLOS",
            "BlinkRate",
            "YawnRate",
            "YawnCount",
            "KSS",
            "Timestamp",
        ])
        writer.writerows(rows_all)
    print(f"\n🎯 Saved combined CSV with all subjects: {output_csv_all}")
else:
    print("\n⚠️ No data extracted. Check your input files!")
