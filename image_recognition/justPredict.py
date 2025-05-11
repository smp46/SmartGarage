import sys
import os
import configparser
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

torch.set_float32_matmul_precision('high')

def predict(device, transform, model, image_path, classes):
    class0, class1 = classes
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return class0 if predicted.item() == 1 else class1

def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    weights = MobileNet_V3_Small_Weights.DEFAULT
    transform = weights.transforms()
    
    # Load Pre-trained MobileNetV3-Large and Modify It
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)  # Initialize MobileNetV3
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )

    model_path = f'./{model_name}.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)

    return device, transform, model

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['DEFAULT']

# Main function modified to handle command-line arguments
def main():
    if open('config.ini', 'r').read() == False:
        print('Please run binaryTrainer.py first to create a model and configuration file.')
        sys.exit(1)
    
    if len(sys.argv) != 2:
        # Request input parameters for prediction
        image_path = input("Enter the path to the image for prediction: ")
    elif len(sys.argv) == 2:
        image_path = sys.argv[1]
    
    config = load_config()
    device, transform, model = load_model(config['ModelName'])
    
    # Assume predict function exists and handles these parameters
    prediction = predict(device, transform, model, image_path, config['Classes'].replace(" ", "").split(','))
    print(prediction)

if __name__ == '__main__':
    main()

