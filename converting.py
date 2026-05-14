import json
import os

with open("YOLO_MODEL/train/_annotations.coco.json") as f:
    data = json.load(f)
    
os.makedirs("labels/train", exist_ok=True)

images = {
    img['id']: img for img in data['images']
}

for ann in data["annotations"]:
    img = images[ann['image_id']]
    w_img, h_img = img['width'], img['height']
    
    x_min, y_min, w_box, h_box = ann['bbox']
    
    x_center = (x_min + w_box / 2) / w_img
    y_center = (y_min + h_box / 2) / h_img
    
    width = 