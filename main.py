import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks import python
import numpy as np
import cv2 as cv
import time
import os
# from ultralytics import YOLO

# https://arxiv.org/pdf/2302.09657 

total_frames = 0;

dataset = []
latest_result = None;
frame = 1;
ball_tracking = True;

# 1.8 KB per frame

# Video capture

# if int(input("Do you want to track the ball? 1-Yes, 2-No: ")) == 2:
#     ball_tracking == False

# if ball_tracking:
#     ball_model = YOLO("runs/detect/train-3/weights/best.pt")

selected_capture_method = int(input("Select a capture method (1-Live Camera, 2-Recorded Video): "))

if selected_capture_method == 1:
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
else:
    videos = []
    num = 1
    for file in os.listdir("videos"):
        if file.endswith((".mp4", ".m4a", ".MOV")):
            videos.append(file)
            print(f"{num}. {file}")
            num+=1
    video = int(input("Select a video by typing the file number: "))
    video_path = f"videos/{videos[video-1]}"
    cap = cv.VideoCapture(video_path)
    # frame_width = int(cap.get(3))  # 3 is cv2.CAP_PROP_FRAME_WIDTH 
    # frame_height = int(cap.get(4)) # 4 is cv2.CAP_PROP_FRAME_HEIGHT
    # out = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc(*'mp4v'), 60, (frame_width, frame_height))

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
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def on_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result



# def process_frame(frame, landmarker, dataset):
#     global latest_result
#     # 1. OPTIMIZATION: Eliminate array copying. Use explicit conversion for MediaPipe.
#     rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
#     if selected_capture_method == 1:
#         timestamp_ms = int(time.time() * 1000)
#         landmarker.detect_async(mp_image, timestamp_ms)
#     else:
#         timestamp_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))
#         latest_result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
#     if ball_tracking:
#         results = ball_model.predict(frame, conf=0.3, verbose=False)
#         ball_x, ball_y = -1.0, -1.0
#         if len(results[0].boxes) > 0:
#             box = results[0].boxes.xywh[0]
#             ball_x, ball_y = float(box[0]), float(box[1])
#             cv.circle(frame, (int(ball_x), int(ball_y)), 10, (0,255,0), -1)
        
#     # If the chosen method is live video, use the result callback and do the rest
#     if latest_result and latest_result.pose_landmarks:
#         for pose_landmarks in latest_result.pose_landmarks:
#             drawing_utils.draw_landmarks( # Draws all 33 landmarks
#                 image=frame,
#                 landmark_list=pose_landmarks,
#                 connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
#                 landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
#                 connection_drawing_spec=drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=4)
#             )
            
#             all_current_landmarks = np.array([])
#             for landmark in pose_landmarks:
#                 all_current_landmarks = np.append(all_current_landmarks, [landmark.x, landmark.y, landmark.z])
#             if (dataset.size > 0):
#                 dataset = np.vstack((dataset, all_current_landmarks))
#             else:
#                 dataset = all_current_landmarks
#     return frame, dataset
    

# landmarker = vision.PoseLandmarker.create_from_options(options)

# dataset = np.array([])

        
def draw_selected_landmarks(frame, pose_landmarks_list):
    """Draw all connections faintly, then highlight tracked joints in bright colour."""
    for pose_landmarks in pose_landmarks_list:
        # Draw full skeleton faintly using the tasks drawing_utils
        drawing_utils.draw_landmarks( # Draws all 33 landmarks
            image=frame,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec=drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=4)
        )

if selected_capture_method == 1:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=on_result,
        num_poses=1
    )
elif selected_capture_method == 2:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )

with PoseLandmarker.create_from_options(options) as landmarker:
    frame_idx = 0
    last_result = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
        frame_idx = int(cap.get(cv.CAP_PROP_POS_FRAMES))

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.pose_landmarks:
                last_result = result

        # Always draw the most recent result (works during pause too)
        if last_result and last_result.pose_landmarks:
            draw_selected_landmarks(frame, last_result.pose_landmarks)
            
        cv.imshow("Analysis", frame)

        #  Key handling 
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

landmarker.close()
cap.release()

# if selected_capture_method == 2:
#     out.release() # This saves the mp4
    
cv.destroyAllWindows()

np.savetxt("pose_landmarks.csv", dataset, delimiter=",", fmt="%1.16f", comments="")