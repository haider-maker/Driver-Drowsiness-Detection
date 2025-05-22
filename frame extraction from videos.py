import cv2
import os

def extract_frames_from_video(video_path, output_folder, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    subject_folder = os.path.join(output_folder, video_name)
    os.makedirs(subject_folder, exist_ok=True)

    frame_index = 0
    saved_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save every 5th frame to reduce redundancy
        if frame_index % frame_skip == 0:
            frame_filename = os.path.join(subject_folder, f"frame_{saved_index:04}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_index += 1

        frame_index += 1

    cap.release()
    print(f"Extracted {saved_index} frames from {video_name}.")
    
videos_dir = './videos'     # where your .mp4 files are
output_dir = './frames'     # where frames will be saved

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(videos_dir):
    if file.endswith('.avi') or file.endswith('.mp4'):
        video_path = os.path.join(videos_dir, file)
        extract_frames_from_video(video_path, output_dir)
