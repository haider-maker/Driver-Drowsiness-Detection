import cv2
import dlib
import os
import math

# === Setup ===
predictor_path = "./models/shape_predictor_68_face_landmarks.dat"
image_dir = "D:\drozy-dataset\Driver-Drowsiness-Detection\data\cropped_faces"
output_dir = "../data/features_output_dlib"
os.makedirs(output_dir, exist_ok=True)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# === Indices ===
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
MOUTH = [48, 54, 51, 62, 66, 57]

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def compute_ear(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def compute_mar(mouth):
    A = euclidean(mouth[2], mouth[4])
    B = euclidean(mouth[3], mouth[5])
    C = euclidean(mouth[0], mouth[1])
    return (A + B) / (2.0 * C)

# === Process ===
for fname in sorted(os.listdir(image_dir)):
    if not fname.endswith((".jpg", ".png")):
        continue

    path = os.path.join(image_dir, fname)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        continue

    rects = detector(image, 1)
    if not rects:
        print(f"No face in {fname}")
        continue

    shape = predictor(image, rects[0])
    coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

    left_eye = [coords[i] for i in LEFT_EYE]
    right_eye = [coords[i] for i in RIGHT_EYE]
    mouth = [coords[i] for i in MOUTH]

    ear = (compute_ear(left_eye) + compute_ear(right_eye)) / 2.0
    mar = compute_mar(mouth)

    # Save to .txt
    out_path = os.path.join(output_dir, fname.replace(".jpg", ".txt"))
    with open(out_path, "w") as f:
        f.write(f"{ear:.6f} {mar:.6f}\n")
