import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np
import cv2 as cv

# Locate the model path
model_path = "pose_landmarker_full.task"

# Landmark Point Color
landmark_color = 255,0,0
landmark_thickness = 2

# Set variables for the options
baseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=baseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

# Drawing landmarks on each frame
def draw_landmarks_on_frame(rgb_image, detection_result):
    # list of landmarks
    pose_landmarks_list = detection_result.pose_landmarks 
    
    # Copies the numpy array image so the model can annotate on it
    annotated_image = np.copy(rgb_image) 
    
    # Selects the style of points (maybe customize?)
    pose_landmark_style = drawing_utils.get_default_pose_landmarks_style()
    
    # Selects the style of connections (customize?)
    pose_connection_style = drawing_utils.DrawingSpec(color=(landmark_color), thickness=landmark_thickness)
    
    # Draws each point in the landmark list
    for pose_landmarks in pose_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style
        )
    return annotated_image # Returns the frame with all annotations


# Video capture
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("can't recieve frame (stream end?). Exiting...")
        break
    # cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
    mp_image = mp.image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)

cap.release()
cv.destroyAllWindows()



# def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
#     print('pose landmarker result: {}'.format(result))
    


# with PoseLandmarker.create_from_options(options) as landmarker:
    