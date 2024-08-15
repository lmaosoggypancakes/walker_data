from ultralytics import YOLO
from roboflow import Roboflow
# Load a model
model = YOLO("yolov8n.pt")
      
model.train(data="data.yaml", epochs=50)
