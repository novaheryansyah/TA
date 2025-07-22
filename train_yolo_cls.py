from ultralytics import YOLO

model = YOLO("yolov8s-cls.pt")  

model.train(
    data='dataset_beras_split',
    epochs=50,
    imgsz=224,
    batch=16,
    project="beras_clf",
    name="yolov8-cls",
    pretrained=True
)

model.val()
model.export(format='onnx') 
