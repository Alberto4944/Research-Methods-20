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
        
k# Draws major points for the targeted joints
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

print(f"\nSaved {rows_written} frames to '{output_csv}'")
print(f"  {STROKE_LABEL} : {len(stroke_frames)}")
print(f"  other          : {rows_written - len(stroke_frames)}")