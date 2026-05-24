"""
label_strokes.py — Table Tennis Stroke Labeling Tool
------------------------------------------------------
Usage:
    python label_strokes.py

Controls:
    S         — Mark START of a forehand drive stroke
    E         — Mark END of a forehand drive stroke (saves segment)
    D         — Delete last saved segment (undo)
    SPACE     — Pause / Resume
    Q         — Quit and save all labeled data to labeled_dataset.csv

Place your videos in a folder called "videos/" next to this script.
Output: labeled_dataset.csv — only the joints defined in joints.py are saved.
"""

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
import os
import csv
from joints import JOINT_INDICES, JOINT_NAMES, FEATURE_COLS

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

STROKE_LABEL = "forehand_drive"
OTHER_LABEL  = "other"

MODEL_PATH = "models/pose_landmarker_lite.task"  # same models folder as main.py

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
video_path = "videos/IMG_1773.MOV"

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
    """Draw all connections faintly, then highlight tracked joints in bright colour."""
    h, w = frame.shape[:2]

    for pose_landmarks in pose_landmarks_list:
        # Draw full skeleton faintly using the tasks drawing_utils
        drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=drawing_utils.DrawingSpec(color=(60, 60, 60), thickness=1, circle_radius=1),
            connection_drawing_spec=drawing_utils.DrawingSpec(color=(60, 60, 60), thickness=1)
        )
        # Overdraw the joints we actually care about in bright yellow/white
        for idx in JOINT_INDICES:
            lm = pose_landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv.circle(frame, (cx, cy), 7, (0, 200, 255), -1)
            cv.circle(frame, (cx, cy), 7, (255, 255, 255), 2)

#  oseLandmarker options (VIDEO mode — synchronous, no callback needed) 
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

# Main loop 
with PoseLandmarker.create_from_options(options) as landmarker:
    frame_idx = 0
    last_result = None
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break
            frame_idx = int(cap.get(cv.CAP_PROP_POS_FRAMES))

        # Pose detection — skip when paused (timestamps must be monotonically increasing)
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

        # Always draw the most recent result (works during pause too)
        if last_result and last_result.pose_landmarks:
            draw_selected_landmarks(frame, last_result.pose_landmarks)

        #  HUD overlay 
        h, w = frame.shape[:2]

        # Progress bar
        progress = frame_idx / max(total_frames, 1)
        cv.rectangle(frame, (0, h - 12), (w, h), (30, 30, 30), -1)
        cv.rectangle(frame, (0, h - 12), (int(w * progress), h), (0, 200, 80), -1)

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

# Write CSV 
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
print(f"  forehand_drive : {len(stroke_frames)}")
print(f"  other          : {rows_written - len(stroke_frames)}")