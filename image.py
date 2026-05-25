import cv2 as cv
import os
SELECT_BALL = False

selected = int(input("Pose or Ball Estimation (pose-1, ball-2)): "))
if selected == 1:
    SELECT_BALL = False
elif selected == 2:
    SELECT_BALL = True;

images = []
num = 1
for file in os.listdir("images"):
    if file.endswith((".jpg", ".jpeg", ".png")):
        images.append(file)
        print(f"{num}. {file}")
        num+=1
image = int(input("Select a image by typing the file number: "))
image_path = f"images/{images[image-1]}"

frame = cv.imread(image_path)
if SELECT_BALL:
    from ultralytics import YOLO
    ball_model = YOLO("runs/detect/train-3/weights/best.pt")

    results = ball_model.predict(frame, conf=0.3, verbose=False)
    ball_x, ball_y = -1.0, -1.0
    if len(results[0].boxes) > 0:
        box = results[0].boxes.xywh[0]
        ball_x, ball_y = float(box[0]), float(box[1])
        cv.circle(frame, (int(ball_x), int(ball_y)), 30, (0,255,0), -1)
        
    cv.imwrite("output.jpg", frame)

if not SELECT_BALL:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import numpy as np
    from mediapipe.tasks.python.vision import drawing_utils
    from mediapipe.tasks.python.vision import drawing_styles
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    def draw_landmarks_on_image(rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
        pose_connection_style = drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=5)

        for pose_landmarks in pose_landmarks_list:
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=pose_landmark_style,
                connection_drawing_spec=pose_connection_style)

        return annotated_image

    # STEP 2: Create an PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path='models/pose_landmarker_full.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        running_mode=VisionRunningMode.IMAGE)
    detector = vision.PoseLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(image_path)

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    cv.imwrite("output.jpg", cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))
    
print("Finished!")