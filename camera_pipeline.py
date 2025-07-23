import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String 
from cv_bridge import CvBridge
import cv2

import json
import dlib
import numpy as np
import math
from collections import deque
import os
from ament_index_python.packages import get_package_share_directory

# Facial Landmark Indices (68-point format)
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
MOUTH = [48, 54, 51, 62, 66, 57]  # [left, right, top_outer, top_inner, bottom_inner, bottom_outer]

class CameraPipeline(Node):
    def __init__(self):
        super().__init__('camera_pipeline')
        
        # Subscriber
        self.image_subscriber = self.create_subscription(
            Image,
            '/synced_output',
            self.image_callback,
            10
        )
        
        # Publisher for extracted features
        self.publisher = self.create_publisher(
            String,
            '/feature_camera',
            10
        )
        
        self.bridge = CvBridge()
        
        # Load face detection models
        self.load_face_models()
        
        # Initialize feature extraction
        self.initialize_feature_extraction()
        
        self.get_logger().info("Drowsiness feature extractor initialized")

    def load_face_models(self):
        """Initialize face detection and landmark models"""
        # Check if the model file exists
        
        # Initialize dlib models
        predictor_path = r"/home/user2/ros2_ws/src/online_feature_pkg/models/shape_predictor_68_face_landmarks.dat"
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(predictor_path)
        
        # Initialize CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        self.get_logger().info("Loaded face detection models")

    def initialize_feature_extraction(self):
        """Initialize feature extraction parameters and buffers"""
        # Parameters
        self.fps = 20.0  # Frames per second
        self.window_size_sec = 30.0
        self.stride_sec = 10.0
        self.window_size = int(self.window_size_sec * self.fps)
        self.stride = int(self.stride_sec * self.fps)
        
        # Thresholds
        self.EAR_THRESHOLD = 0.25
        self.MAR_THRESHOLD = 0.42
        
        # Buffers
        self.ear_buffer = deque(maxlen=self.window_size)
        self.mar_buffer = deque(maxlen=self.window_size)
        self.timestamp_buffer = deque(maxlen=self.window_size)
        
        # State variables
        self.frame_counter = 0
        self.last_publish_frame = 0
        self.eye_state = 'open'  # 'open' or 'closed'
        self.mouth_state = 'closed'  # 'open' or 'closed'
        self.last_blink_end = -999
        self.last_yawn_end = -999
        self.blink_count = 0
        self.yawn_count = 0
        
        # Constants
        self.MIN_BLINK_GAP = 6  # frames
        self.MIN_YAWN_GAP = 30  # frames

    def image_callback(self, msg):
        try:
            image= msg.image
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
            
            # Preprocess image
            gray = self.preprocess_image(cv_image)
            
            # Detect faces
            faces = self.detect_faces(gray)
            if not faces:
                self.get_logger().warn("No faces detected")
                return
                
            # Use first detected face
            main_face = faces[0]
            
            # Get facial landmarks
            landmarks = self.get_landmarks(gray, main_face)
            
            # Calculate EAR and MAR
            ear, mar = self.calculate_features(landmarks)
            
            # Update buffers
            self.update_buffers(ear, mar, msg.header.stamp)
            
            # Update blink/yawn states
            self.update_eye_state(ear)
            self.update_mouth_state(mar)
            
            # Process window periodically
            self.process_window(msg.header.stamp)
            
        except Exception as e:
            self.get_logger().error(f"Processing failed: {str(e)}")

    def preprocess_image(self, image):
        """Preprocess image for better detection"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply CLAHE for contrast enhancement
        gray = self.clahe.apply(gray)
        
        # Apply Gaussian blur
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        return gray

    def detect_faces(self, gray_image):
        """Detect faces using dlib"""
        # Try with standard detection
        faces = self.face_detector(gray_image, 0)
        
        # If no faces found, try with upsampling
        if not faces:
            faces = self.face_detector(gray_image, 1)
            
        return faces

    def get_landmarks(self, gray_image, face_rect):
        """Get facial landmarks using dlib"""
        shape = self.landmark_predictor(gray_image, face_rect)
        return np.array([[p.x, p.y] for p in shape.parts()])

    def calculate_features(self, landmarks):
        """Calculate EAR and MAR from landmarks"""
        # EAR calculation
        left_eye = [landmarks[i] for i in LEFT_EYE]
        right_eye = [landmarks[i] for i in RIGHT_EYE]
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        # MAR calculation
        mouth = [landmarks[i] for i in MOUTH]
        mar = self.calculate_mar(mouth)
        
        return ear, mar

    def calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio (EAR)"""
        A = self.euclidean(eye_points[1], eye_points[5])
        B = self.euclidean(eye_points[2], eye_points[4])
        C = self.euclidean(eye_points[0], eye_points[3])
        return (A + B) / (2.0 * C)

    def calculate_mar(self, mouth_points):
        """Calculate Mouth Aspect Ratio (MAR)"""
        A = self.euclidean(mouth_points[2], mouth_points[4])  # top_outer to bottom_outer
        B = self.euclidean(mouth_points[3], mouth_points[5])  # top_inner to bottom_inner
        C = self.euclidean(mouth_points[0], mouth_points[1])  # left to right
        return (A + B) / (2.0 * C)

    def euclidean(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def update_buffers(self, ear, mar, timestamp):
        """Update feature buffers with current values"""
        self.ear_buffer.append(ear)
        self.mar_buffer.append(mar)
        self.timestamp_buffer.append(timestamp)
        self.frame_counter += 1

    def update_eye_state(self, ear):
        """Update eye state and count blinks"""
        if self.eye_state == 'open' and ear < self.EAR_THRESHOLD:
            # Check if enough time has passed since last blink
            if self.frame_counter - self.last_blink_end > self.MIN_BLINK_GAP:
                self.blink_count += 1
            self.eye_state = 'closed'
        elif self.eye_state == 'closed' and ear >= self.EAR_THRESHOLD:
            self.eye_state = 'open'
            self.last_blink_end = self.frame_counter

    def update_mouth_state(self, mar):
        """Update mouth state and count yawns"""
        if self.mouth_state == 'closed' and mar > self.MAR_THRESHOLD:
            # Check if enough time has passed since last yawn
            if self.frame_counter - self.last_yawn_end > self.MIN_YAWN_GAP:
                self.yawn_count += 1
            self.mouth_state = 'open'
        elif self.mouth_state == 'open' and mar <= self.MAR_THRESHOLD:
            self.mouth_state = 'closed'
            self.last_yawn_end = self.frame_counter

    def process_window(self, timestamp):
        """Process the current window and publish features if needed"""
        # Check if it's time to process a new window
        if self.frame_counter - self.last_publish_frame < self.stride:
            return
            
        # Check if we have enough data
        if len(self.ear_buffer) < self.window_size:
            self.get_logger().warn(f"Not enough frames for window: {len(self.ear_buffer)}/{self.window_size}")
            return
            
        # Calculate features
        perclos = self.calculate_perclos()
        blink_rate = self.blink_count / self.window_size_sec
        yawn_rate = self.yawn_count / self.window_size_sec
        
        # Publish features
        self.publish_features(perclos, blink_rate, yawn_rate, timestamp)
        
        # Reset counters for next window
        self.blink_count = 0
        self.yawn_count = 0
        self.last_publish_frame = self.frame_counter

    def calculate_perclos(self):
        """Calculate PERCLOS (percentage of eye closure)"""
        closed_frames = sum(1 for ear in self.ear_buffer if ear < self.EAR_THRESHOLD)
        return closed_frames / len(self.ear_buffer)

    def publish_features(self, perclos, blink_rate, yawn_rate, timestamp):
        """Publish extracted features as JSON in a std_msgs/String."""
        # pack features into a JSON-compatible dict
        payload = {
            'stamp': timestamp.sec + timestamp.nanosec * 1e-9,
            'perclos': round(perclos, 4),
            'blink_rate': round(blink_rate, 4),
            'yawn_rate': round(yawn_rate, 4)
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.publisher.publish(msg)
        self.get_logger().info(
            f"Published features → {msg.data}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CameraPipeline()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
