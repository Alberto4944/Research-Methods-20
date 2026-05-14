import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks import python
import numpy as np
import cv2 as cv
import time
import os

dataset = []
latest_result = None;
frame = 1;

# 1.8 KB per frame

# Video capture

selected_capture_method = int(input("Select a capture method (1-Live Camera, 2-Recorded Video): "))

if selected_capture_method == 1:
    cap = cv.VideoCapture(0)
else:
    videos = []
    num = 1
    for file in os.listdir("videos"):
        if file.endswith((".mp4", ".m4a", ".MOV")):
            videos.append(file)
            print(f"{num}. {file}")
            num+=1
    video = int(input("Select a video by typing the file number: "))
    cap = cv.VideoCapture(f"videos/{videos[video-1]}")
    frame_width = int(cap.get(3))  # 3 is cv2.CAP_PROP_FRAME_WIDTH 
    frame_height = int(cap.get(4)) # 4 is cv2.CAP_PROP_FRAME_HEIGHT
    out = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc(*'mp4v'), 60, (frame_width, frame_height))

selected_model = int(input("Select a Model (1-lite, 2-full, 3-heavy): "))

if selected_model == 1:
    model_path = "models/pose_landmarker_lite.task"
elif selected_model == 2:
    model_path = "models/pose_landmarker_full.task"
elif selected_model == 3: 
    model_path = "models/pose_landmarker_heavy.task"

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

if selected_capture_method == 1:
    options = PoseLandmarkerOptions(
        base_options=baseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=on_result,
        num_poses=1
    )
elif selected_capture_method == 2:
    options = PoseLandmarkerOptions(
        base_options=baseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

def process_frame(frame, landmarker, dataset):
    global latest_result
    # Define the RGB and Mediapipe images, using the input frame
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Capture Specific Timestamps and detection methods
    if selected_capture_method == 1:
        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)
    else:
        timestamp_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))
        latest_result = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    # Duplicates the RGB frame to be annotated on and returned back
    annotated = np.copy(rgb_frame)
        
    # If the chosen method is live video, use the result callback and do the rest
    if latest_result and latest_result.pose_landmarks:
        for pose_landmarks in latest_result.pose_landmarks:
            drawing_utils.draw_landmarks( # Draws all 33 landmarks
                image=annotated,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
                connection_drawing_spec=drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=4)
            )
            all_current_landmarks = np.array([])
            for landmark in pose_landmarks:
                all_current_landmarks = np.append(all_current_landmarks, [landmark.x, landmark.y, landmark.z])
            if (dataset.size > 0):
                dataset = np.vstack((dataset, all_current_landmarks))
            else:
                dataset = all_current_landmarks
    return cv.cvtColor(annotated, cv.COLOR_RGB2BGR), dataset

landmarker = vision.PoseLandmarker.create_from_options(options)

dataset = np.array([])

if selected_capture_method == 1:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("can't recieve frame (stream end?). Exiting...")
            break
        frame, dataset = process_frame(frame, landmarker, dataset)
        cv.imshow("Pose Estimation", cv.flip(frame, 1))
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

if selected_capture_method == 2:
    if not cap.isOpened():
        print("Error: Cannot open video file")
        exit()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame, dataset = process_frame(frame, landmarker, dataset)
        out.write(frame)
        cv.imshow("Pose Estimation", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

landmarker.close()
cap.release()

if selected_capture_method == 2:
    out.release() # This saves the mp4
    
cv.destroyAllWindows()

np.savetxt("pose_landmarks.csv", dataset, delimiter=",", fmt="%1.16f", comments="")