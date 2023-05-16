from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
LOAD_MODEL_FILE = "/opt/ml/baseline/yolov8/trash_split/88epoch/weights/best.pt"
model = YOLO(LOAD_MODEL_FILE)# Use the model

model.val(data="recycle.yaml", imgsz = 1024, mode="val")  # valid the model