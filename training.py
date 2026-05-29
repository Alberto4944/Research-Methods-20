from ultralytics import YOLO

model = YOLO("yolo11n.pt") 

results = model.train(
    data="dataset/data.yaml", # This points to the folder Roboflow created
    epochs=40,      # 50 is usually enough for a small dataset
    imgsz=640,     # Standard resolution
    plots=True,      
    resume=True
)