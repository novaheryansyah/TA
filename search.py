from ultralytics import YOLO
model = YOLO("beras_clf/yolov8-cls/weights/best.pt")
print(model.names)
