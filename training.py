from ultralytics import YOLO

model = YOLO("yolo11n.pt") 

# 2. Train the model on your custom data
# 'data' should point to the data.yaml file inside your downloaded Roboflow folder
results = model.train(
    data="dataset/data.yaml", # This points to the folder Roboflow created
    epochs=50,      # 50 is usually enough for a small dataset
    imgsz=640,     # Standard resolution
    plots=True,      # This saves charts like 'results.png' for your slides!
    resume=True
)

# 3. Export the model
# model.export(format="onnx")