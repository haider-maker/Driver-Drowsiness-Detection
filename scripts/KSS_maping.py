import os
from pathlib import Path

# === CONFIG ===
kss_file_path = Path("D:/DROZY/KSS.txt")
frames_base_dir = Path("./frames")
output_extension = ".kss"

# === Read and parse KSS values ===
kss_scores = []
with open(kss_file_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            kss_scores.append([int(p) for p in parts])

# Flatten KSS with mapping like:
# Row 0: [1-1, 1-2, 1-3]
# Row 1: [2-1, 2-2, 2-3]
kss_map = {}  # e.g., "1-1": 3
for subject_idx, (v1, v2, v3) in enumerate(kss_scores, start=1):
    kss_map[f"{subject_idx}-1"] = v1
    kss_map[f"{subject_idx}-2"] = v2
    kss_map[f"{subject_idx}-3"] = v3

# === Assign KSS to frames ===
for session_dir in sorted(frames_base_dir.iterdir()):
    if not session_dir.is_dir():
        continue

    session_name = session_dir.name
    if session_name not in kss_map:
        print(f"⚠️ No KSS for session {session_name}")
        continue

    kss_value = str(kss_map[session_name])
    frame_files = sorted(session_dir.glob("frame_*.jpg"))
    for frame_file in frame_files:
        kss_path = frame_file.with_suffix(output_extension)
        with open(kss_path, "w") as kf:
            kf.write(kss_value)

    print(f"✅ Mapped KSS {kss_value} to {len(frame_files)} frames in {session_name}")

print("\n🎉 All KSS values mapped successfully.")
