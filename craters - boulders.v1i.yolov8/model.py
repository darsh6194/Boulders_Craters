from ultralytics import YOLO
import torch

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load a pre-trained YOLOv8 model (YOLOv8n is the nano version; use YOLOv8s, YOLOv8m for larger models)
model = YOLO("yolov8n.yaml")  # Initialize a new YOLOv8 model

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Train the model
model.train(
    data=r"D:\Projects\Boulders And Craters\craters - boulders.v1i.yolov8\data.yaml",  # Path to data.yaml
    epochs=50,                # Number of training epochs
    imgsz=640,                # Image size (default: 640x640)
    batch=16,                 # Batch size
    workers=4,
    device=device                # Number of dataloader workers
    )

# Validate the model on the validation set
metrics = model.val()  # Evaluate performance on validation data
print("Validation Results:", metrics)

# Test the model on the test set
results = model.test(data=r"D:\Projects\Boulders And Craters\craters - boulders.v1i.yolov8\data.yaml")  # Evaluate performance on the test dataset
print("Test Results:", results)

# Predict on new images or datasets
predict_results = model.predict(
    source="test/images",  # Path to test images
    save=True,                    # Save predictions to disk
    conf=0.25                     # Confidence threshold
)
print("Predictions Complete!")

# Export the trained model for deployment
model.export(format="onnx")  # You can export to other formats like TorchScript, CoreML, etc.
