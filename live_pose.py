import cv2 as cv
import mediapipe as mp
import time
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

# Initialize MediaPipe shortcuts
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variable to safely hold the most recent background frame results
latest_result = None

def on_result(result, output_image, timestamp_ms):
    """Callback function that receives data from the background thread."""
    global latest_result
    latest_result = result

# Define asynchronous pipeline settings using the lightweight tracking profile
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/pose_landmarker_full.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=on_result,
    num_poses=1
)

# Open hardware webcam connection (0 is typically the built-in front camera)
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # Enforce buffer capacity limit to prevent input latency lag

with PoseLandmarker.create_from_options(options) as landmarker:
    print("[i] Live pipeline successfully initialized. Press 'q' to exit stream.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Standardize and scale frame down to 960 width baseline to save processing power
        h, w = frame.shape[:2]
        if w != 960:
            frame = cv.resize(frame, (960, int(h * 960 / w)))

        # Flip horizontally for a natural mirror effect during presentations
        frame = cv.flip(frame, 1)

        # 2. Convert from native OpenCV BGR format into standard model RGB space
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 3. Asynchronously offload frame calculations to keep the display loop fast
        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        # 4. Corrected drawing utility logic matching your project's main.py framework
        # 4. Corrected drawing utility logic matching your project's main.py framework
        if latest_result and latest_result.pose_landmarks:
            for pose_landmarks in latest_result.pose_landmarks:
                drawing_utils.draw_landmarks( # Draws all 33 landmarks
                image=frame,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
                connection_drawing_spec=drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=4)
                )

        # Render display window output
        cv.imshow("Live Biomechanical Tracking", frame)
        
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv.destroyAllWindows()