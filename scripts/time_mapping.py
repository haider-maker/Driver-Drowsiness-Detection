import os

# === CONFIG ===
timestamps_dir = "D:/DROZY/timestamps"      # Folder containing .txt timestamp files like 1-1.txt
frames_base_dir = "./frames"                # Your full frames folder with subfolders like 1-1, 1-2, etc.
skip_factor = 5                              # You extracted every 5th frame from the video

# === MAIN PROCESSING ===
for ts_filename in os.listdir(timestamps_dir):
    if not ts_filename.endswith(".txt"):
        continue

    session_name = ts_filename.replace(".txt", "")
    ts_path = os.path.join(timestamps_dir, ts_filename)
    frames_dir = os.path.join(frames_base_dir, session_name)

    if not os.path.exists(frames_dir):
        print(f"⚠️ Skipping {session_name}: no matching folder in frames/")
        continue

    # Load timestamps and format as HH:MM:SS:MS
    with open(ts_path, 'r') as f:
        lines = f.readlines()

    timestamps = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 7:
            hh, mm, ss, ms = parts[3], parts[4], parts[5], parts[6]
            formatted_ts = f"{hh}:{mm}:{ss}:{ms}"
            timestamps.append(formatted_ts)

    # Assign timestamps to every 5th frame
    frame_idx = 0
    while True:
        frame_filename = f"frame_{frame_idx:04}.jpg"
        frame_path = os.path.join(frames_dir, frame_filename)
        if not os.path.exists(frame_path):
            break  # No more frames

        ts_line_index = frame_idx * skip_factor
        if ts_line_index >= len(timestamps):
            print(f"⚠️ Not enough timestamps for {session_name}, frame {frame_idx}")
            break

        timestamp = timestamps[ts_line_index]

        # Write .time file
        time_filename = f"frame_{frame_idx:04}.time"
        time_path = os.path.join(frames_dir, time_filename)
        with open(time_path, 'w') as tf:
            tf.write(f"{timestamp}\n")

        frame_idx += 1

    print(f"✅ Processed timestamps for: {session_name}")

print("\n🎉 All timestamps mapped to frames.")
