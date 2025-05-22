import subprocess

# === CONFIG ===
model = "yolov8n.pt"
data_yaml = "data.yaml"
epochs = 50
imgsz = 640
batch = 16

# === BUILD COMMAND ===
command = [
    "yolo", 
    "task=detect", 
    "mode=train",
    f"model={model}",
    f"data={data_yaml}",
    f"epochs={epochs}",
    f"imgsz={imgsz}",
    f"batch={batch}"
]

# === RUN TRAINING ===
try:
    print("🚀 Starting YOLOv8 training...")
    subprocess.run(command, check=True)
    print("✅ Training completed successfully.")
except subprocess.CalledProcessError as e:
    print("❌ Training failed with error:")
    print(e)
