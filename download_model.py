from ultralytics import YOLO
import os
import shutil

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Download YOLOv5x model
model = YOLO("yolov5x.pt")

# Save the complete model
target_model = os.path.join("models", "best.pt")
model.save(target_model)
print(f"Model successfully saved to {target_model}")

# Clean up any temporary files
if os.path.exists("yolov5x.pt"):
    os.remove("yolov5x.pt")

# Clean up the dataset directory if it exists
if os.path.exists("football-players-detection-12"):
    shutil.rmtree("football-players-detection-12") 