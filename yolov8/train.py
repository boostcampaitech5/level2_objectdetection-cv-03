from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt") 
# Use the model
model.train(data="recycle.yaml", imgsz = 1024)  # train the model