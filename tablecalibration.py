import cv2 as cv
import numpy as np

TABLE_WIDTH_CM = 274.0
TABLE_LENGTH_CM = 152.5

TABLE_IRL_CORNERS = np.array([
    [0,              0],               # top-left
    [TABLE_WIDTH_CM, 0],               # top-right
    [TABLE_WIDTH_CM, TABLE_LENGTH_CM], # bottom-right
    [0,              TABLE_LENGTH_CM], # bottom-left
], dtype=np.float32)

class CalibrateTable:
    def __init__(self):
        self.points = []
        self.homography = None
        self.is_calibrated = False
        self.in_calibration_mode = False
        self.frozen_frame = None
    
    def start_calibrating(self, frame):
        self.points = []
        self.is_calibrated = False
        self.in_calibration_mode = True
        self.frozen_frame = frame.copy()
        self.homography = None
        
    def handle_click(self, event, x, y, flags, param):
        if event != cv.EVENT_LBUTTONDOWN or not self.in_calibration_mode:
            return
        if len(self.points) >= 4:
            return
        self.points.append([x, y])
        if len(self.points) == 4:
            self.find_homography() # Runs homography when we have 4 points
    
    def find_homography(self):
        source = np.array(self.points, dtype=np.float32)
        self.homography, mask = cv.findHomography(source, TABLE_IRL_CORNERS)
        self.is_calibrated = True
        self.in_calibration_mode = False
        
    def convert_pixel_to_cm(self, px, py):
        if not self.is_calibrated:
            return None
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        out = cv.perspectiveTransform(pt, self.homography)
        return float(out[0][0][0]), float(out[0][0][1])
    
    def draw(self, frame):
        h, w = frame.shape[:2]
        
        if self.in_calibration_mode:
            frame[:] = self.frozen_frame
            
            labels = ["Top Left", "Top Right", "Bottom Right", "Bottom Left"]
            colors = [(0, 255, 255), (0, 200, 255), (0, 165, 255), (0, 100, 255)]
            for i in range(len(self.points)):
                point_x, point_y = self.points[i]
                cv.circle(frame, (point_x, point_y), 8, colors[i], -1) # Filled circle
                cv.circle(frame, (point_x, point_y), 8, (255, 255, 255), 2)
                cv.putText(frame, labels[i], (point_x + 10, point_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
            
            for i in range(len(self.points)-1):
                cv.line(frame, self.points[i], self.points[i+1], (0, 255, 255), 1)
            if len(self.points) == 4:
                cv.line(frame, self.points[3], self.points[0], (0, 255, 255), 1)
            remaining = 4 - len(self.points)
            msg = f"Click {remaining} more corner{'s' if remaining != 1 else ''} (TL TR BR BL)"
            cv.rectangle(frame, (0, h-40), (w, h), (0,0,0), -1)
            cv.putText(frame, msg, (10, h-12),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                
        elif self.is_calibrated:
            # Draw the calibrated table outline
            pts = np.array(self.points, dtype=np.int32)
            cv.polylines(frame, [pts], isClosed=True, color=(0,255,100), thickness=2)
            cv.putText(frame, "Table calibrated", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,100), 2)
            
        return frame