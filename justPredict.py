import sys
import os
import configparser
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.transforms import RandomApply, ColorJitter, GaussianBlur

# Define the Dataset Class
class GarageDoorDataset(Dataset):
    def __init__(self, base_path, transform, classes):
        class0, class1 = classes
        self.transform = transform
        # Load images and labels
        self.data = []
        for label, folder in enumerate([class0, class1]):
            folder_path = os.path.join(base_path, folder)
            images = glob.glob(f'{folder_path}/*.jpg')  # Assuming images are in .jpg format
            for image in images:
                self.data.append((image, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image, label

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
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model_path = f'./{model_name}.pth'
    model.load_state_dict(torch.load(model_path))
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