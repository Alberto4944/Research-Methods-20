import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np
import cv2 as cv
import time

one = 0;

latest_result = None;

# Locate the model path
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

# Drawing landmarks on each frame

# Video capture
cap = cv.VideoCapture(0)
landmarker = vision.PoseLandmarker.create_from_options(options)


if not cap.isOpened():
    print("Cannot open camera")
    exit()
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
                    color=(0, 230, 118), thickness=2
                ),
            )
            if one == 0:
                print(pose_landmarks[0].x)
                one+=1
                numpy_array = ([])
                for landmark in pose_landmarks: 
                    numpy_array.append(landmark.x)
                np.savetxt("pose_landmarks.csv", numpy_array, delimiter=",", fmt="%d")
                
        frame = cv.cvtColor(annotated, cv.COLOR_RGB2BGR)

    cv.imshow("Pose Estimation", cv.flip(frame, 1))
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
landmarker.close()
cap.release()
cv.destroyAllWindows()
