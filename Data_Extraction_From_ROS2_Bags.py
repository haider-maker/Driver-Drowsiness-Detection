import os
import csv
import cv2
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from cv_bridge import CvBridge

# === CONFIGURATION ===
bag_path = '/Users/yashraj/Library/CloudStorage/OneDrive-TechnischeHochschuleIngolstadt/THI/Academics/Sem 4/Summer Project/LLM-based Agent for Driver Sleepiness Detection and Mitigation in Automotive Systems/Dataset/ROSBAGS/Fabian_KSS_8_Vid_1/Fabian_KSS_8_Vid_1_0.db3'
output_folder = 'extracted_data'
image_folder = os.path.join(output_folder, 'images')
os.makedirs(image_folder, exist_ok=True)

bridge = CvBridge()

# === ROS 2 BAG SETUP ===
reader = SequentialReader()
storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
converter_options = ConverterOptions('', '')
reader.open(storage_options, converter_options)

# Get topic-type mapping
topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
type_map = {}

# Data containers
data_rows = []
image_rows = []

print("🔄 Reading bag file...")

# === DATA EXTRACTION LOOP ===
while reader.has_next():
    topic, data, t = reader.read_next()
    if topic not in type_map:
        type_map[topic] = get_message(topic_types[topic])
    msg_type = type_map[topic]
    msg = deserialize_message(data, msg_type)

    timestamp = t / 1e9  # Convert from nanoseconds to seconds

    # Parse steering/offset data
    if topic == '/data_capture/data':
        try:
            parsed = msg.data
            parts = parsed.split(',')
            steering = float(parts[0].split(':')[1].strip())
            offset = float(parts[1].split(':')[1].strip())
            data_rows.append({'timestamp': timestamp, 'steering': steering, 'offset': offset})
        except Exception as e:
            print(f"⚠️ Skipped malformed message: {parsed} -> {e}")

    # Save camera image and track metadata
    elif topic == '/camera/image_raw':
        try:
            image_filename = f"{timestamp:.3f}.png"
            filepath = os.path.join(image_folder, image_filename)
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imwrite(filepath, cv_img)
            image_rows.append({'timestamp': timestamp, 'image_filename': image_filename})
        except Exception as e:
            print(f"⚠️ Image error at {timestamp:.3f}: {e}")

# === WRITE CSV OUTPUTS ===

# Steering & offset data
csv_file = os.path.join(output_folder, 'data_capture.csv')
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['timestamp', 'steering', 'offset'])
    writer.writeheader()
    writer.writerows(data_rows)

# Image metadata (timestamp + filename)
image_csv_file = os.path.join(image_folder, 'image_metadata.csv')
with open(image_csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['timestamp', 'image_filename'])
    writer.writeheader()
    writer.writerows(image_rows)

print("✅ Extraction complete.")
print(f"📝 Steering data CSV saved at: {csv_file}")
print(f"🖼️ {len(image_rows)} images saved in: {image_folder}")
print(f"📄 Image metadata CSV saved at: {image_csv_file}")
