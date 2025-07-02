from pathlib import Path
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
from collections import defaultdict

# === CONFIGURATION ===
model_path = "shape_predictor_68_face_landmarks.dat"
input_dir = Path("data/Fabian_KSS_8_Vid_1")
fps = 20
window_size_sec = 120
stride_sec = 30
window_size = window_size_sec * fps
stride = stride_sec * fps
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.42
MIN_BLINK_GAP = 6
MIN_YAWN_GAP = 30
output_csv = "final_features_prototype.csv"
visual_output_dir = Path("prototype_visuals")
visual_output_dir.mkdir(parents=True, exist_ok=True)

# === Load dlib models ===
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

# === Helper functions ===
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_ear(landmarks):
    def eye_aspect_ratio(eye):
        A = euclidean(eye[1], eye[5])
        B = euclidean(eye[2], eye[4])
        C = euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)
    left_eye = [landmarks[i] for i in range(36, 42)]
    right_eye = [landmarks[i] for i in range(42, 48)]
    return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

def compute_mar(landmarks):
    A = euclidean(landmarks[61], landmarks[67])
    B = euclidean(landmarks[62], landmarks[66])
    C = euclidean(landmarks[63], landmarks[65])
    D = euclidean(landmarks[60], landmarks[64])
    return (A + B + C) / (3.0 * D)

def count_blinks(ears):
    blink_count = 0
    blinking = False
    last_blink_end = -999
    for idx, ear in enumerate(ears):
        if ear < EAR_THRESHOLD and not blinking and idx - last_blink_end > MIN_BLINK_GAP:
            blink_count += 1
            blinking = True
        elif ear >= EAR_THRESHOLD and blinking:
            blinking = False
            last_blink_end = idx
    return blink_count

def count_yawns(mars):
    yawn_count = 0
    yawning = False
    last_yawn_end = -999
    for idx, mar in enumerate(mars):
        if mar > MAR_THRESHOLD and not yawning and idx - last_yawn_end > MIN_YAWN_GAP:
            yawn_count += 1
            yawning = True
        elif mar <= MAR_THRESHOLD and yawning:
            yawning = False
            last_yawn_end = idx
    return yawn_count

# === Process all frames ===
video_id = input_dir.name
frames = sorted(input_dir.glob("frame_*.jpg"))
total_frames = len(frames)

print(f"Found {total_frames} frames in the folder.")

all_frame_numbers = []
processed_frame_numbers = []
ear_all = []
mar_all = []
all_data = []
random_visuals = []

for frame_path in frames:
    # Extract frame number from filename
    frame_num = int(frame_path.stem.split("_")[1])
    all_frame_numbers.append(frame_num)

    img = cv2.imread(str(frame_path))
    if img is None:
        print(f"Could not read image {frame_path}. Skipping.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    if len(dets) == 0:
        # No face detected
        continue

    shape = predictor(gray, dets[0])
    landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    ear = compute_ear(landmarks)
    mar = compute_mar(landmarks)
    ear_all.append(ear)
    mar_all.append(mar)
    processed_frame_numbers.append(frame_num)

    all_data.append({
        "video": video_id,
        "frame": frame_path.name,
        "ear": ear,
        "mar": mar
    })

    if len(random_visuals) < 3 and random.random() < 0.1:
        img_vis = img.copy()
        for (x, y) in landmarks:
            cv2.circle(img_vis, (x, y), 1, (0, 255, 0), -1)
        save_path = visual_output_dir / f"{video_id}_{frame_path.stem}.jpg"
        cv2.imwrite(str(save_path), img_vis)
        random_visuals.append(save_path)

# === Check for missing frames ===
missing = set(all_frame_numbers) - set(processed_frame_numbers)

print(f"\nTotal frames in folder       : {total_frames}")
print(f"Frames successfully processed: {len(processed_frame_numbers)}")
print(f"Frames with no face detected : {len(missing)}")

if len(missing) > 0:
    missing_sorted = sorted(missing)
    print("Missing frame numbers (no face detected):")
    print(missing_sorted)

# === Plot EAR & MAR graph ===
if len(ear_all) > 0:
    plt.figure(figsize=(12,5))
    plt.plot(processed_frame_numbers, ear_all, label="EAR", color="blue")
    plt.plot(processed_frame_numbers, mar_all, label="MAR", color="orange")
    plt.xlabel("Frame number")
    plt.ylabel("EAR / MAR")
    plt.title("EAR and MAR over processed frames")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(visual_output_dir / "ear_mar_plot.png")
    plt.show()
else:
    print("No EAR/MAR data extracted — skipping plot.")

# === Windowed Feature Extraction ===
video_data = defaultdict(list)
for item in all_data:
    video_data[item["video"]].append(item)

rows = []
for video, frames in video_data.items():
    frames_sorted = sorted(frames, key=lambda x: int(x["frame"].split("_")[1].split(".")[0]))
    total_frames_processed = len(frames_sorted)
    print(f"\nTotal frames processed in video {video}: {total_frames_processed}")

    if total_frames_processed < window_size:
        print("Not enough frames for window. Processing all frames as one window.")
        ears = [f["ear"] for f in frames_sorted]
        mars = [f["mar"] for f in frames_sorted]
        if len(ears) > 0:
            perclos = sum(1 for e in ears if e < EAR_THRESHOLD) / len(ears)
            blink_rate = count_blinks(ears) / (len(ears)/fps)
            yawn_rate = count_yawns(mars) / (len(ears)/fps)
            yawn_count = count_yawns(mars)
            rows.append([
                video,
                frames_sorted[len(frames_sorted)//2]["frame"],
                f"{perclos:.4f}",
                f"{blink_rate:.4f}",
                f"{yawn_rate:.4f}",
                yawn_count
            ])
    else:
        for i in range(0, total_frames_processed - window_size + 1, stride):
            window = frames_sorted[i:i + window_size]
            ears = [f["ear"] for f in window]
            mars = [f["mar"] for f in window]
            if len(ears) != window_size:
                continue
            perclos = sum(1 for e in ears if e < EAR_THRESHOLD) / len(ears)
            blink_rate = count_blinks(ears) / window_size_sec
            yawn_rate = count_yawns(mars) / window_size_sec
            yawn_count = count_yawns(mars)
            rows.append([
                video,
                window[len(window)//2]["frame"],
                f"{perclos:.4f}",
                f"{blink_rate:.4f}",
                f"{yawn_rate:.4f}",
                yawn_count
            ])

# === Save CSV ===
if rows:
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Video", "Frame", "PERCLOS", "BlinkRate", "YawnRate", "YawnCount"])
        writer.writerows(rows)
    print(f"\nSaved CSV with {len(rows)} rows.")
else:
    print("\nNo data written to CSV. Check your inputs or window size.")
