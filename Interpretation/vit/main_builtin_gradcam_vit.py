import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
import os 
from tqdm import tqdm
from datetime import datetime
import pandas as pd 

def load_and_preprocess_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Convert to numpy and normalize for displaying
    rgb_img = np.array(image)
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255.0
    return rgb_img, input_tensor

def get_model(model_name, use_cuda):
    model = torch.hub.load('facebookresearch/deit:main', model_name, pretrained=True)
    model.eval()
    if use_cuda:
        model = model.cuda()
    return model

def get_cam_method(method_name, model, target_layers):
    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
    }
    if method_name == "ablationcam":
        return methods[method_name](model=model, target_layers=target_layers, ablation_layer=AblationLayerVit())
    else:
        return methods[method_name](model=model, target_layers=target_layers,
                                    reshape_transform=lambda x: x[:, 1:].reshape(x.size(0), 14, 14, -1).permute(0, 3, 1, 2))

def generate_and_save_cam(image_path, model_name, method_name, img_name, saved_path, use_cuda=True, aug_smooth=False, eigen_smooth=False):
    model = get_model(model_name, use_cuda)
    target_layers = [model.blocks[-1].norm1]
    cam = get_cam_method(method_name, model, target_layers)

    rgb_img, input_tensor = load_and_preprocess_image(image_path)
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor, targets=None, eigen_smooth=eigen_smooth, aug_smooth=aug_smooth)
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    output_filename = f'{method_name}_{model_name}_{image_path.split("/")[-1]}'

    output_filename = os.path.join(saved_path, f'{img_name}_{method_name}.jpg')

    cv2.imwrite(output_filename, cam_image)
    # print(f'Saved {output_filename}')

def main():
    model_name = 'deit_tiny_patch16_224'
    methods = ['gradcam', 'scorecam', 'gradcam++']
    
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

    img_type = 'original'

    # Loop over all img_names
    for img_name in tqdm(img_names, desc="Processing Images"):
        img_path = os.path.join(data_dir, img_type, img_name)


        for method in methods:
            generate_and_save_cam(img_path, model_name, method, img_name, saved_path, use_cuda=True, aug_smooth=False, eigen_smooth=False)


if __name__ == '__main__':
    main()
    u=1

