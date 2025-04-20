from ultralytics import YOLO

# Load the YOLOv8 model configuration or pretrained weights
model = YOLO("yolov8m.pt")  # You can switch to yolov8n.pt for a lighter model

# Path to the dataset configuration YAML file
path = "dataset/data.yaml"  # Update with correct relative path if needed

# Train the model
results = model.train(data=path, epochs=10)

# Evaluate the model
results = model.val()

# Export the model to ONNX format
success = model.export(format="onnx")
