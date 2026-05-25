import cv2 as cv
from ultralytics import YOLO

frame = cv.imread("wintergames.jpg")
ball_model = YOLO("runs/detect/train-3/weights/best.pt")

results = ball_model.predict(frame, conf=0.3, verbose=False)
ball_x, ball_y = -1.0, -1.0
if len(results[0].boxes) > 0:
    box = results[0].boxes.xywh[0]
    ball_x, ball_y = float(box[0]), float(box[1])
    cv.circle(frame, (int(ball_x), int(ball_y)), 30, (0,255,0), -1)
    
cv.imwrite("tested.jpg", frame)