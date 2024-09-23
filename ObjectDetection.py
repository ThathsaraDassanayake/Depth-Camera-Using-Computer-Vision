
from roboflow import Roboflow

# Set the path for images and model weights
sample_image_path = "/Users/sasindu/Desktop/Depth Camera/CameraCalibration/left.jpg"
new_test_image_path = "/Users/sasindu/Desktop/Depth Camera/CameraCalibration/right.jpg"

# Initialize YOLOv8 model (hypothetical API usage)
from ultralytics import YOLO  # Assuming ultralytics provides a direct import for YOLOv8

# Initialize Roboflow
rf = Roboflow(api_key="3IDDvO8wLwsxDOg32ia9")
project = rf.workspace("sandaru-o5zia").project("bottle_detector-lnu6c")
version = project.version(1)
dataset = version.download("yolov8-obb")

# Train the model
#data_yaml = os.path.join(dataset.location, 'data.yaml')
train_model = YOLO('yolov8m.pt')
results= train_model.train(data="/Users/sasindu/Desktop/DC/bottle_detector-1/data.yaml", epochs=20, imgsz=640)

