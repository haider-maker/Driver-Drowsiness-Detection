# LLM-based Agent for Driver Sleepiness Detection and Mitigation in Automotive Systems

This repository contains the complete modular implementation of a feature extraction pipeline using rule-based features and ROS2 integration. The project was part of an academic course, where we achieved all planned milestones including offline feature extraction, dataset recording, and live deployment.

## Branch Structure Overview

### 'main'
contains readme file explaining each branch and final .ppt 


### '02-camera'
Handles the "offline camera pipeline" using the DROZY IR dataset and self-recorded grayscale video:
- Facial landmark detection using d-lib.
- EAR and MAR computation.
- Time-windowed feature extraction: PERCLOS, Blink Rate, Yawn Count.
- Different regression and classification models for fatigue score estimation

### '03-Carla'
Implements "offline vehicle feature extraction" from the CARLA simulator:
- Collects steering and lateral offset at 20Hz.
- Computes entropy, SRR, lane-keeping, and lane departure events.
- Uses a ROS2 node to process streamed vehicle data.

### '04-ros-data'
Contains the "ROS2 pipeline for vehicle stream (Carla)":
- Sync the data for the camera and the Carla at 20Hz 
- Subscribes to CARLA vehicle telemetry.
- Uses a sliding window to publish real-time feature vectors.
- Supports integration with the fatigue classifier node.

### '05-Fatigue-classifier'
Includes the "rule-based classifier" for both camera and vehicle features:
- Uses the majority vote approach for the fatigue score estimation
- Aggregates fatigue indicators (PERCLOS, SRR, etc.).
- Outputs fatigue levels or drowsiness scores.


### '06-feature-nodes'
Implements "three ROS2 nodes":
- /features_camera: For live inference on IR grayscale feed.
- /features_carla: For vehicle dynamics features.
- /fatigue_estimator: Combines both inputs and publishes final fatigue score.


