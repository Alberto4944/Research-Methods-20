import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

from mediapipe.tasks import python

import numpy as np
import cv2 as cv
import time

model_path = ""

frame = 1;

one = 0;

# 1.8 KB per frame

dataset = []

latest_result = None;
3

selected_model = int(input("Select a Model (1-lite, 2-full, 3-heavy): "))

if selected_model == 1:
    model_path = "pose_landmarker_lite.task"

elif selected_model == 2:
    model_path = "pose_landmarker_full.task"

elif selected_model == 3: 
    model_path = "pose_landmarker_heavy.task"

# Landmark Point Color
landmark_color = 255,0,0
landmark_thickness = 2

# Set variables for the options
baseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def on_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = PoseLandmarkerOptions(
    base_options=baseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=on_result,
    num_poses=1
)

# Video capture
cap = cv.VideoCapture(1)
landmarker = vision.PoseLandmarker.create_from_options(options)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

dataset = np.array([])

while True:
    ret, frame = cap.read()
    if not ret:
        print("can't recieve frame (stream end?). Exiting...")
        break
    
    
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    timestamp_ms = int(time.time() * 1000)
    landmarker.detect_async(mp_image, timestamp_ms)
        
    # Draw the most recent result
    if latest_result and latest_result.pose_landmarks:
        annotated = np.copy(rgb_frame)
        for pose_landmarks in latest_result.pose_landmarks:
            drawing_utils.draw_landmarks(
                image=annotated,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
                connection_drawing_spec=drawing_utils.DrawingSpec(
                    color=(255, 0, 0), thickness=4
                ),
            )
            
            all_current_landmarks = np.array([])
            for landmark in pose_landmarks:
                all_current_landmarks = np.append(all_current_landmarks, [landmark.x, landmark.y, landmark.z])
            
            if (dataset.size > 0):
                dataset = np.vstack((dataset, all_current_landmarks))
            else:
                dataset = all_current_landmarks
                
        frame = cv.cvtColor(annotated, cv.COLOR_RGB2BGR)

    cv.imshow("Pose Estimation", cv.flip(frame, 1))
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
landmarker.close()
cap.release()
cv.destroyAllWindows()

np.savetxt("pose_landmarks.csv", dataset, delimiter=",", fmt="%1.16f", comments="")