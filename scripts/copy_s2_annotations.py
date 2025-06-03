import os
import shutil
from pathlib import Path

# === CONFIG ===
s2_root = Path("data/frames_mapped_s2")       # Folder with subfolders like 1-1, 1-2, ...
cropped_dir = Path("data/cropped_faces")      # Where cropped IR face images live
output_suffix = "_face0.txt"                  # Expected output format

copied = 0
skipped = 0

for cropped_file in cropped_dir.glob("*_face0.jpg"):
    base = cropped_file.stem.replace("_face0", "")  # e.g., 1-1_frame_3066
    try:
        video_id, frame_id = base.split("_frame_")
        s2_txt = s2_root / video_id / f"frame_{frame_id}.txt"

        if s2_txt.exists():
            out_txt = cropped_dir / f"{base}_face0.txt"
            shutil.copy(s2_txt, out_txt)
            copied += 1
        else:
            print(f"❌ Annotation not found for {base}")
            skipped += 1
    except Exception as e:
        print(f"⚠️ Failed to process {cropped_file.name}: {e}")
        skipped += 1

print(f"\n✅ Done. Copied {copied} annotation files. Skipped {skipped}.")
