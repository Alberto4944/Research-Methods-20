<<<<<<< HEAD
import customtkinter as ctk
import subprocess
import webbrowser
import os
import sys

P5_VIEWER_URL = "https://your-username.github.io/your-repo"

ctk.set_appearance_mode("dark")
=======
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks import python

import numpy as np
import cv2 as cv
import time
import os
import csv
import pandas as pd

import customtkinter as ctk
import os

from ultralytics import YOLO

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

P5_VIEWER_URL = "https://your-username.github.io/your-repo"


# MAIN.PY
def launch_main():
    # https://arxiv.org/pdf/2302.09657 

    total_frames = 0;

    dataset = []
    latest_result = None;
    frame = 1;
    ball_tracking = True;

    # set this based on your camera setup
    VIEW = "front"  # or "front"


    JOINT_INDICES, JOINT_NAMES, FEATURE_COLS = launch_joints()

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


    selected_model = int(input("Select a Model (1-lite, 2-full, 3-heavy): "))

    if selected_model == 1:
        model_path = "models/pose_landmarker_lite.task"
    elif selected_model == 2:
        model_path = "models/pose_landmarker_full.task"
    elif selected_model == 3: 
        model_path = "models/pose_landmarker_heavy.task"
        
    classifier = joblib.load("stroke_classifier.pkl") if os.path.exists("stroke_classifier.pkl") else None
    if classifier:
        print("Classifier loaded")
    else:
        print("[i] No classifier found — running pose only")

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
        last_result = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break

            h, w = frame.shape[:2]
            if w > 960:
                frame = cv.resize(frame, (960, int(h * 960 / w)))

            if selected_capture_method == 1:
                frame = cv.flip(frame, 1)

            rgb      = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            if selected_capture_method == 1:
                landmarker.detect_async(mp_image, int(time.time() * 1000))
                last_result = latest_result  # from callback
            else:
                timestamp_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                if result.pose_landmarks:
                    last_result = result

            if last_result and last_result.pose_landmarks:
                draw_selected_landmarks(frame, last_result.pose_landmarks)
                
                # Classifier
                if classifier:
                    lm_flat = []
                    
                    for idx in JOINT_INDICES:
                        lm = last_result.pose_landmarks[0][idx]
                        lm_flat += [lm.x, lm.y, lm.z]
                    label = classifier.predict([lm_flat])[0]
                    prob  = max(classifier.predict_proba([lm_flat])[0])
                    
                    color = (0, 200, 80) if label == "forehand_drive" else (180, 180, 180)
                    cv.putText(frame, f"{label} {prob:.0%}", (10, 120),
                            cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv.imshow("Analysis", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv.destroyAllWindows()

# JOINTS.PY
def launch_joints():
    # (index, name) — for readability and CSV column naming
    RELEVANT_JOINTS = [
        (0,  "nose"),               # head/body orientation proxy

        (11, "left_shoulder"),      # body rotation reference
        (12, "right_shoulder"),     # swing arm root

        (13, "left_elbow"),         # body rotation reference  
        (14, "right_elbow"),        # swing arm hinge

        (15, "left_wrist"),         # opposite side balance
        (16, "right_wrist"),        # racket hand — most important

        (18, "right_pinky"),        # racket grip finish position
        (20, "right_index"),        # racket grip finish position

        (23, "left_hip"),           # weight transfer + stance
        (24, "right_hip"),          # weight transfer + stance
    ]

    # Just the indices, for easy slicing
    JOINT_INDICES = [j[0] for j in RELEVANT_JOINTS]
    JOINT_NAMES   = [j[1] for j in RELEVANT_JOINTS]

    # Build flat CSV column names: lm_nose_x, lm_nose_y, lm_nose_z, ...
    FEATURE_COLS = []
    for name in JOINT_NAMES:
        FEATURE_COLS += [f"lm_{name}_x", f"lm_{name}_y", f"lm_{name}_z"]
        
    return JOINT_INDICES, JOINT_NAMES, FEATURE_COLS

# IMAGE.PY
def launch_image():
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

# TRAIN-CLASSIFIER.PY
def launch_train_classifier():
    DATASET = "labeled_dataset.csv"
    OUTPUT  = "stroke_classifier.pkl"

    # Load
    if not os.path.exists(DATASET):
        print(f"[!] {DATASET} not found — run classify.py first to label some strokes")
        exit()

    print("Loading dataset...")
    df = pd.read_csv(DATASET)

    # Show counts for every label in the dataset dynamically
    labels = df["label"].unique().tolist()
    print(f"  Total : {len(df)} frames")
    for l in labels:
        print(f"  {l} : {len(df[df['label'] == l])}")

    # Warn if any stroke class is too small but don't block training
    for l in labels:
        if l != "other" and len(df[df["label"] == l]) < 30:
            print(f"[!] Warning: only {len(df[df['label'] == l])} frames for '{l}' — accuracy may be low")

    # Features and labels
    X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy()

    # 80/20 train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining on {len(X_train)} frames, testing on {len(X_test)}...")

    # Train
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, max_depth=20, min_samples_split=10, class_weight="balanced")
    model.fit(X_train, y_train)

    # Results
    y_pred = model.predict(X_test)
    print("\n--- Accuracy ---")
    print(classification_report(y_test, y_pred))

    # Confusion matrix — dynamic based on whatever classes exist
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    header = f"{'':20}" + "".join(f"{l:20}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = f"  {row_label:18}" + "".join(f"{cm[i][j]:<20}" for j in range(len(labels)))
        print(row)

    # Feature importance
    print("\n--- Top 5 Most Important Joints ---")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(min(5, len(FEATURE_COLS))):
        idx = indices[i]
        print(f"  {FEATURE_COLS[idx]:<25} {importances[idx]:.4f}")

    # Save
    joblib.dump(model, OUTPUT)
    print(f"\n[✓] Saved to {OUTPUT}")
    print("Now run main.py — it will load the classifier automatically")

# CLASSIFY.PY
def launch_classifier():

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    strokes = {
        1: "forehand_drive",
        2: "backhand_drive", 
        3: "forehand_push",
        4: "backhand_push",
        5: "forehand_loop",
    }

    print("Select stroke to label:")
    for key in strokes.keys():
        print(f"{key}: {strokes.get(key)}")

    STROKE_LABEL = strokes[int(input("Stroke: "))]
    OTHER_LABEL  = "other"

    MODEL_PATH = "models/pose_landmarker_full.task"  # same models folder as main.py

    # Pick video
    videos_dir = "videos"
    print("Avaliable Videos:")
    videos = []
    num = 1
    for file in os.listdir(videos_dir):
        if file.endswith((".mp4", ".m4a", ".MOV")):
            videos.append(file)
            print(f"{num}. {file}")
            num+=1
    choice = int(input("Select a video number")) - 1
        
    video_path = os.path.join(videos_dir, videos[choice])

    # Output CSV 
    output_csv = "labeled_dataset.csv"
    already_exists = os.path.exists(output_csv)
    csv_header = ["frame", "video"] + FEATURE_COLS + ["label"]

    # State
    all_frames_data = {}   # frame_number -> flat list of selected landmark values
    segments = []          # list of (start_frame, end_frame) marked as STROKE_LABEL
    stroke_start = None
    paused = False

    cap = cv.VideoCapture(video_path)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS)

    print(f"\nLoaded: {videos[choice]} | {total_frames} frames @ {fps:.1f} fps")
    print(f"Saving {len(JOINT_INDICES)} joints ({len(FEATURE_COLS)} features) per frame")
    print("Controls: S=Start  E=End  D=Undo  SPACE=Pause  Q=Save+Quit\n")

    # Draw helper 
    def draw_selected_landmarks(frame, pose_landmarks_list):
        # Draws a faint skeleton for the entire body
        h, w = frame.shape[:2]
        for pose_landmarks in pose_landmarks_list:
            drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=drawing_utils.DrawingSpec(color=(60, 60, 60), thickness=1, circle_radius=1),
                connection_drawing_spec=drawing_utils.DrawingSpec(color=(60, 60, 60), thickness=1)
            )
            
            # Draws major points for the targeted joints
            for index in JOINT_INDICES:
                lm = pose_landmarks[index]
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                cv.circle(frame, (cx, cy), 7, (0, 200, 255), -1)
                cv.circle(frame, (cx, cy), 7, (255, 255, 255), 2)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )

    # Main loop 
    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        last_result = None
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video.")
                    break
                frame_idx = int(cap.get(cv.CAP_PROP_POS_FRAMES))

            if not paused:
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks:
                    last_result = result
                    lm_flat = []
                    for idx in JOINT_INDICES:
                        lm = result.pose_landmarks[0][idx]
                        lm_flat += [lm.x, lm.y, lm.z]
                    all_frames_data[frame_idx] = lm_flat

            if last_result and last_result.pose_landmarks:
                draw_selected_landmarks(frame, last_result.pose_landmarks)

            h, w = frame.shape[:2]

            cv.putText(frame, f"Frame {frame_idx}/{total_frames}", (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(frame, f"Segments saved: {len(segments)}", (10, 60),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 120), 2)

            if stroke_start is not None:
                cv.putText(frame, f"[RECORDING from frame {stroke_start}]", (10, 90),
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
                cv.rectangle(frame, (0, 0), (w - 1, h - 13), (0, 0, 220), 4)

            if paused:
                cv.putText(frame, "PAUSED", (w // 2 - 60, h // 2),
                        cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3)

            cv.putText(frame, "S=Start  E=End  D=Undo  SPACE=Pause  Q=Save+Quit",
                    (10, h - 20), cv.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv.imshow("Stroke Labeler", frame)

            #  Key handling 
            key = cv.waitKey(1 if not paused else 50) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('s'):
                stroke_start = frame_idx
                print(f"  [S] Stroke START at frame {frame_idx}")
            elif key == ord('e'):
                if stroke_start is None:
                    print("  [!] Press S first to mark the start.")
                elif frame_idx <= stroke_start:
                    print("  [!] End frame must be after start frame.")
                else:
                    segments.append((stroke_start, frame_idx))
                    print(f"  [E] Segment ({stroke_start} → {frame_idx}) saved")
                    stroke_start = None
            elif key == ord('d'):
                if segments:
                    print(f"  [D] Removed segment {segments.pop()}")
                else:
                    print("  [D] Nothing to undo.")

    cap.release()
    cv.destroyAllWindows()

    if not all_frames_data:
        print("No pose data collected. Did MediaPipe detect anyone in the video?")
        exit()

    stroke_frames = set()
    for (start, end) in segments:
        for f in range(start, end + 1):
            stroke_frames.add(f)

    rows_written = 0
    with open(output_csv, "a" if already_exists else "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not already_exists:
            writer.writerow(csv_header)
        for frame_num in sorted(all_frames_data.keys()):
            label = STROKE_LABEL if frame_num in stroke_frames else OTHER_LABEL
            writer.writerow([frame_num, videos[choice]] + all_frames_data[frame_num] + [label])
            rows_written += 1

    print(f"\n✓ Saved {rows_written} frames to '{output_csv}'")
    print(f"  {STROKE_LABEL} : {len(stroke_frames)}")
    print(f"  other          : {rows_written - len(stroke_frames)}")

# FEEDBACK.PY
def launch_feedback():
    # ── Math helpers ──────────────────────────────────────────────────────────────

    def calc_angle(a, b, c):
        """Calculate the angle at point b formed by a->b->c, in degrees."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def landmark_to_point(lm, dims=2):
        """Convert a MediaPipe landmark to a numpy array. dims=2 for x/y, 3 for x/y/z."""
        if dims == 3:
            return np.array([lm.x, lm.y, lm.z])
        return np.array([lm.x, lm.y])

    def normalize_distance(lm_a, lm_b):
        """Return the distance between two landmarks — used as body scale reference."""
        a = landmark_to_point(lm_a)
        b = landmark_to_point(lm_b)
        return np.linalg.norm(a - b) + 1e-6

    # ── Feedback rules ────────────────────────────────────────────────────────────
    # Each rule returns a string if the issue is detected, or None if form is good.
    # Thresholds are normalized relative to shoulder width so they work for any body size.

    # SIDE VIEW RULES
    # Camera is to the left or right of the player.
    # Good for: elbow extension, forward swing arc, weight transfer front/back.

    def check_elbow_angle_side(landmarks, scale):
        r_shoulder = landmark_to_point(landmarks[12])
        r_elbow    = landmark_to_point(landmarks[14])
        r_wrist    = landmark_to_point(landmarks[16])
        angle = calc_angle(r_shoulder, r_elbow, r_wrist)
        # Ideal forehand drive elbow angle at contact: 120-160 degrees
        if angle < 110:
            return "Bend your elbow less — arm too cramped"
        if angle > 170:
            return "Bend your elbow more — arm too straight"
        return None

    def check_wrist_height_side(landmarks, scale):
        r_shoulder = landmark_to_point(landmarks[12])
        r_wrist    = landmark_to_point(landmarks[16])
        # Wrist should be below shoulder at contact (positive y = lower on screen)
        diff = (r_wrist[1] - r_shoulder[1]) / scale
        if diff < -0.3:
            return "Lower your wrist — too high at contact"
        return None

    def check_forward_lean_side(landmarks, scale):
        nose      = landmark_to_point(landmarks[0])
        l_hip     = landmark_to_point(landmarks[23])
        r_hip     = landmark_to_point(landmarks[24])
        hip_mid_x = (l_hip[0] + r_hip[0]) / 2
        # Nose should be ahead of hips (smaller x if facing right)
        diff = (nose[0] - hip_mid_x) / scale
        if abs(diff) < 0.1:
            return "Lean into the shot — transfer your weight forward"
        return None

    # FRONT VIEW RULES
    # Camera is in front of or behind the player.
    # Good for: hip rotation, shoulder symmetry, lateral wrist position.

    def check_hip_rotation_front(landmarks, scale):
        l_hip = landmark_to_point(landmarks[23])
        r_hip = landmark_to_point(landmarks[24])
        # On a forehand drive hips should rotate — right hip forward means r_hip.x < l_hip.x
        diff = (l_hip[0] - r_hip[0]) / scale
        if diff < 0.1:
            return "Rotate your hips — turn into the shot"
        return None

    def check_shoulder_rotation_front(landmarks, scale):
        l_shoulder = landmark_to_point(landmarks[11])
        r_shoulder = landmark_to_point(landmarks[12])
        diff = (l_shoulder[0] - r_shoulder[0]) / scale
        if diff < 0.05:
            return "Rotate your shoulders — follow through more"
        return None

    def check_wrist_position_front(landmarks, scale):
        r_shoulder = landmark_to_point(landmarks[12])
        r_wrist    = landmark_to_point(landmarks[16])
        # Wrist should cross midline on follow through
        diff = (r_shoulder[0] - r_wrist[0]) / scale
        if diff < 0.1:
            return "Follow through across your body more"
        return None

    # ── Main feedback function ────────────────────────────────────────────────────

    SIDE_RULES  = [check_elbow_angle_side, check_wrist_height_side, check_forward_lean_side]
    FRONT_RULES = [check_hip_rotation_front, check_shoulder_rotation_front, check_wrist_position_front]

    def get_feedback(landmarks, view):
        """
        Run all feedback rules for the given camera view.

        Args:
            landmarks : list of MediaPipe NormalizedLandmark (result.pose_landmarks[0])
            view      : "side" or "front"

        Returns:
            list of feedback strings (empty list = good form)
        """
        # Use shoulder width as the body scale reference
        scale = normalize_distance(landmarks[11], landmarks[12])
        rules = SIDE_RULES if view == "side" else FRONT_RULES

        tips = []
        for rule in rules:
            result = rule(landmarks, scale)
            if result:
                tips.append(result)
        return tips


    # ── Quick test ────────────────────────────────────────────────────────────────
    if __name__ == "__main__":
        print("feedback.py loaded ok")
        print(f"Side rules  : {len(SIDE_RULES)}")
        print(f"Front rules : {len(FRONT_RULES)}")
        print("Import get_feedback and pass it a pose_landmarks[0] list to use.")
    
ctk.set_appearance_mode("light")
>>>>>>> a3fc9d9dc80f7601112d1f3d8387ac90931386ee
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("RMD20L - Table Tennis AI")
        self.geometry("500x600")
        self.resizable(False, False)
        
<<<<<<< HEAD
        self.active_processes = []
        self.create_ui()
    def create_ui(self):
=======
>>>>>>> a3fc9d9dc80f7601112d1f3d8387ac90931386ee
        ctk.CTkLabel(self, text="Table Tennis AI", font=ctk.CTkFont(size=28, weight="bold")).pack(pady=(30, 4))
        ctk.CTkLabel(self, text="RMD20L SAGE Project  —  Albert Wu", font=ctk.CTkFont(size=13), text_color="gray").pack(pady=(0, 30))
        
        self.section_header("Data Collection")
        self.run_button(
            "Analyze Results - main.py",
<<<<<<< HEAD
            "View MediaPipe landmarks and YOLOv8 predictions",
            self.launch_main
        )
        self.run_button(
            "Label Strokes - classify.py",
            "Label forehand drive segments to build the training dataset",
            self.launch_classifier
        )
        
        
        
=======
            "View MediaPipe landmarks and YOLOv11n predictions",
            launch_main
        )
        self.run_button(
            "Run mediapipe and YOLO on images - image.py",
            "Draw the key landmarks or the balls location on an image",
            launch_image
        )
        
        self.section_header("Training Models")
        self.run_button(
            "Label Strokes - classify.py",
            "Label stroke segments to build the training dataset",
            launch_classifier
        )
        self.run_button(
            "Train the Classifier - train-classifier.py",
            "Train the random forest ML algorithm on different strokes",
            launch_train_classifier
        )
>>>>>>> a3fc9d9dc80f7601112d1f3d8387ac90931386ee
        
    def section_header(self, text):
        ctk.CTkLabel(self, text=text.upper(), font=ctk.CTkFont(size=11, weight="bold"), text_color="white").pack(anchor="w", padx=24, pady=(10, 2))
    
    def run_button(self, title, subtitle, command, disabled=False): # Only three actual arguments are title, subtitle, and command
        frame = ctk.CTkFrame(self, fg_color=("gray90", "gray17"), corner_radius=10)
        frame.pack(fill="x", padx=20, pady=4)
        
        text_frame = ctk.CTkFrame(frame, fg_color="transparent")
        text_frame.pack(side="left", fill="both", expand=True, padx=14, pady=10)
        
        ctk.CTkLabel(text_frame, text=title, font=ctk.CTkFont(size=14, weight="bold"), anchor="w").pack(anchor="w")
        ctk.CTkLabel(text_frame, text=subtitle, font=ctk.CTkFont(size=11), text_color="grey", anchor="w", wraplength=300).pack(anchor="w")
        
        button = ctk.CTkButton(
            frame,
            text="Run" if not disabled else "Soon", # Checks if currently exists
            width = 70,
            height = 36,
            state="disabled" if disabled else "normal",
            fg_color="grey40" if disabled else None,
            command=command
<<<<<<< HEAD
        ).pack(side="right", padx=14, pady=10)
        

    def run_script(self, name):
        script_path = os.path.join(os.path.dirname(__file__), name)
        # if not os.path.exists(script_path):
        #     self.log(f"[!] Could not find {name}")
        #     return

        # self.log(f"[>] Launching {name}...")
        cwd = os.path.dirname(os.path.abspath(script_path))

        proc = subprocess.Popen( # Had to look online about how to run external programs as well as the tkinter program
            [sys.executable, script_path],
            cwd=cwd
        )
        self.active_processes.append(proc)
        
    def launch_classifier(self):
        self.run_script("classify.py")
    def launch_main(self):
        self.run_script("main.py")
=======
        )
        button.pack(side="right", padx=14, pady=10)
>>>>>>> a3fc9d9dc80f7601112d1f3d8387ac90931386ee

app = App()
app.mainloop()