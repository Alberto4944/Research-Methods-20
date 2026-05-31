import cv2 as cv
import os
from ultralytics import YOLO
from tablecalibration import CalibrateTable
import numpy as np

file_type_choice = 0;
files = []
num = 1


ball_model = YOLO("best_models/best2.pt")

def fix_brightness(frame):
    inv_gamma = 1.0 / 0.8  # adjust this value, lower = darker
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv.LUT(frame, table)

def track_ball(frame):
    h, w = frame.shape[:2]
    if w > 960:
        frame = cv.resize(frame, (960, int(h * 960 / w)))
    if w < 960:
        frame = cv.resize(frame, (960, int(h/w*960)))
    results = ball_model.predict(frame, conf=0.3, verbose=False)
    ball_x, ball_y = -1.0, -1.0
    if len(results[0].boxes) > 0:
        box = results[0].boxes.xywh[0]
        ball_x, ball_y = float(box[0]), float(box[1])
        cv.circle(frame, (int(ball_x), int(ball_y)), 10, (0,255,0), -1)
    return frame, ball_x, ball_y

while file_type_choice != 1 and file_type_choice != 2 and file_type_choice != 3:
    file_type_choice = int(input("Image (1), Video (2), or Live Camera (3): "))

if file_type_choice == 1:
    for file in os.listdir("images"):
        if file.endswith((".jpg", ".jpeg", ".png")):
            files.append(file)
            print(f"{num}. {file}")
            num+=1
    file_choice = int(input("Select an image by typing the file number: "))
    image_path = f"images/{files[file_choice-1]}"
    cv.imshow("Ball Tracking", track_ball(cv.imread(image_path)))
    cv.waitKey(0)

elif file_type_choice == 2:
    for file in os.listdir("videos"):
        if file.endswith((".mp4", ".mov", ".MOV")):
            files.append(file)
            print(f"{num}. {file}")
            num+=1
    file_choice = int(input("Select a video by typing the file number: "))
    video_path = f"videos/{files[file_choice-1]}"
    cap = cv.VideoCapture(video_path)
        
else:
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

if file_type_choice == 2 or file_type_choice == 3:
    cal = CalibrateTable()
    cv.namedWindow("Ball Tracking")
    cv.setMouseCallback("Ball Tracking", cal.handle_click)
    while cap.isOpened():
        ret, frame = cap.read();
        if not ret:
            print("End of video/livestream")
            break
        
        # frame = fix_brightness(frame)

        frame, ball_x, ball_y = track_ball(frame)
        
        if cal.is_calibrated and ball_x != -1.0:
            cm = cal.convert_pixel_to_cm(ball_x, ball_y)
            if cm:
                cv.putText(frame, f"{cm[0]:.1f}cm, {cm[1]:.1f}cm",
                        (int(ball_x) + 12, int(ball_y)),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cal.draw(frame)  # draws table outline or calibration UI
        
        cv.imshow("Ball Tracking", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            cal.start_calibrating(frame)  # freeze and start clicking
    cap.release()
    
cv.destroyAllWindows()