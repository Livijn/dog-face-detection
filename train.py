from pathlib import Path
from ultralytics import YOLO

data = str(Path('./dog_faces_dataset/dog_faces.yaml').resolve())

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data=data,  # path to dataset YAML
    epochs=30,  # number of training epochs
    imgsz=500,  # training image size
    device="mps",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("input_image.jpg")
results[0].show()

# Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model
