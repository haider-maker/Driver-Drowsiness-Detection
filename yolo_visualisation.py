import cv2

# === CONFIG ===
image_path = "./yolo_dataset/images/train/frame_0054.jpg"
label_path = "./yolo_dataset/labels/train/frame_0054.txt"
save_path = "visualized_frame_0000.jpg"  # optional

# === READ IMAGE ===
img = cv2.imread(image_path)
img_height, img_width = img.shape[:2]

# === READ YOLO LABEL ===
with open(label_path, 'r') as f:
    line = f.readline().strip()
    class_id, x_center, y_center, width, height = map(float, line.split())

# === CONVERT TO PIXEL COORDINATES ===
x_center *= img_width
y_center *= img_height
width *= img_width
height *= img_height

x1 = int(x_center - width / 2)
y1 = int(y_center - height / 2)
x2 = int(x_center + width / 2)
y2 = int(y_center + height / 2)

# === DRAW BOUNDING BOX ===
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(img, f"Class {int(class_id)}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# === SHOW & SAVE ===
cv2.imshow("YOLO Label Visualization", img)
cv2.imwrite(save_path, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
