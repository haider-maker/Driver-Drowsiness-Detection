import cv2
import os

# === CONFIG ===
landmark_txt_path = "./frames/1-1/frame_0000.txt"   # Change this
image_path = "./frames/1-1/frame_0000.jpg"          # Change this
output_path = "./frame_0000_landmarks.jpg"


if __name__ == '__main__':
    # === READ LANDMARK COORDINATES ===
    with open(landmark_txt_path, 'r') as f:
        coords = list(map(float, f.read().strip().split()))

    if len(coords) != 136:
        raise ValueError("Expected 68 (x, y) points (136 numbers total), but got something else.")

    # Convert to list of (x, y) tuples
    landmarks = [(int(coords[i]), int(coords[i+1])) for i in range(0, len(coords), 2)]

    # === LOAD IMAGE ===
    img = cv2.imread(image_path)

    # === DRAW LANDMARKS ===
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(img, str(i + 1), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    # === SAVE AND SHOW ===
    cv2.imwrite(output_path, img)
    cv2.imshow("Landmarks", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
