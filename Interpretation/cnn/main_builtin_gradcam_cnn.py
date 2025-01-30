from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, AblationCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import pandas as pd
import os 
from datetime import datetime
from tqdm import tqdm

# Load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

def generate_and_save_cam(model, input_tensor, target_layers, target_class, cam_type, img_name, saved_path):
    if cam_type == 'gradcam':
        cam = GradCAM(model=model, target_layers=target_layers)
    elif cam_type == 'gradcam++':
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    elif cam_type == 'ablationcam':
        cam = AblationCAM(model=model, target_layers=target_layers)
    elif cam_type == 'scorecam':
        cam = ScoreCAM(model=model, target_layers=target_layers)
    else:
        raise ValueError("Unsupported CAM type. Choose 'gradcam', 'gradcam++', 'ablationcam', or 'scorecam'.")

    targets = [ClassifierOutputTarget(target_class)]

    # Generate CAM
    cam_result = cam(input_tensor=input_tensor, targets=targets)

    # Convert CAM result to format that can be overlaid on the RGB image
    cam_result = cam_result[0, :]  # Assume single image batch
    rgb_img = np.array(Image.open(img_path).resize((224, 224)))  # Resize to match input tensor
    rgb_img = np.float32(rgb_img) / 255.0  # Normalize to [0, 1]

    visualization = show_cam_on_image(rgb_img, cam_result, use_rgb=True)

    # Save the visualization
    output_filename = os.path.join(saved_path, f'{img_name}_{cam_type}.jpg')
    cv2.imwrite(output_filename, np.uint8(255 * visualization))

# Load your model
# Replace with your own model loading code if needed
model = resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Get current directory
current_dir = os.getcwd()
# Get the current time
current_time = datetime.now()
# Convert the current time to a string
time_string = current_time.strftime('%Y-%m-%d_%H-%M-%S')

# Go three steps back from the current directory (from current_dir)
dir_3_step_back = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

data_dir = os.path.join(dir_3_step_back, 'data', 'wound_data', 'oneFolder')

# Read CSV file
path_csv = os.path.join(data_dir, 'dbPolicy1.csv')
df = pd.read_csv(path_csv)
img_names = df.iloc[:, 0].values

# Create the full path to the new folder
saved_path = os.path.join(current_dir, 'saved_img', time_string)

# Create the folder if it doesn't exist
if not os.path.exists(saved_path):
    os.makedirs(saved_path)

img_type = 'cropped'

# Loop over all img_names
for img_name in tqdm(img_names, desc="Processing Images"):
    img_path = os.path.join(data_dir, img_type, img_name)

    # Load the image and prepare input tensor
    input_tensor = load_image(img_path)

    # Define target layers and class
    # Adjust target_class according to your trained model
    target_layers = [model.layer4[-1]]
    target_class = 0  # Example target class, change based on your model

    # Generate and save GradCAM result
    generate_and_save_cam(model, input_tensor, target_layers, target_class, 'gradcam', img_name, saved_path)

    # Generate and save GradCAM++ result
    generate_and_save_cam(model, input_tensor, target_layers, target_class, 'gradcam++', img_name, saved_path)

    # Generate and save ScoreCAM result
    generate_and_save_cam(model, input_tensor, target_layers, target_class, 'scorecam', img_name, saved_path)

    u=1
