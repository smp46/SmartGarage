import sys
import os
import configparser
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from tqdm import tqdm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.amp import autocast, GradScaler
from torch.utils.data import random_split
from copy import deepcopy

torch.set_float32_matmul_precision('high')

class GarageDoorDataset(Dataset):
    def __init__(self, base_path, transform, classes):
        class0, class1 = classes
        self.transform = transform
        self.data = []
        for label, folder in enumerate([class0, class1]):
            folder_path = os.path.join(base_path, folder)
            images = glob.glob(f'{folder_path}/*.jpg') 
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

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, early_stop_patience=5):
    model.train()
    scaler = GradScaler()
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

        val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}: Train Acc = {100 * correct / total:.2f}%, Val Acc = {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_wts)


def train_main(classes, num_epochs, model_name, images_path):
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )
    model.to(device)

    weights = MobileNet_V3_Small_Weights.DEFAULT
    transform = weights.transforms()

    # Dataset and Split
    dataset = ImageFolder(images_path, transform=transform)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4, shuffle=False)

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # Train
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs)

    # Save model
    model_path = f'./{model_name}.pth'
    torch.save(model.state_dict(), model_path)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model = torch.compile(model, mode="reduce-overhead")

    return device, transform, model


def validater(device, transform, model, val_base_path, classes):
    val_dataset = GarageDoorDataset(val_base_path, transform, classes)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    # Evaluate the model
    return evaluate(model, val_loader, device)

def evaluate(model, val_loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

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

    model = torch.compile(model, mode="reduce-overhead")

    return device, transform, model

def save_config(model_name, what, classes):
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'ModelName': model_name, 'ClassifiedObject': what, 'Classes': ','.join(classes)}
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['DEFAULT']

def print_usage():
    print("\nUsage: python training.py [action]\n")
    print("Actions:")
    print("  train - Train the model on a dataset of images")
    print("  validate - Test the model accuracy on a validation dataset")
    print("  predict - Classify an image based on the trained model")
    print("  help, h, -h, --help - Display this help message\n")

def main():
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)

    action = sys.argv[1]
    if action not in ['train', 'validate', 'predict', 'help', 'h', '-h', '--help']:
        print("Invalid action.")
        print_usage()
        sys.exit(1)

    if action == 'train':
        image_path = input("Enter the path to the training images: ")
        what = input("What is the object you are trying to classify? ")
        classes = input("Enter the classification names separated by a comma: ").replace(" ", "").split(',')
        model_name = input("Enter the model name to save as: ")
        num_epochs = int(input("Enter the number of epochs: "))
        save_config(model_name, what, classes)
        
        train_main(classes, num_epochs, model_name, image_path)

    elif action == 'validate':
        val_path = input("Enter the path to the validation images: ")
        config = load_config()
        device, transform, model = load_model(config['ModelName'])
        classes = config['Classes'].split(',')

        accuracy = validater(device, transform, model, val_path, classes)
        print(f'The model accuracy is {accuracy}%')

    elif action == 'predict':
        image_path = input("Enter the path to the image for prediction: ")
        config = load_config()
        device, transform, model = load_model(config['ModelName'])
        
        prediction = predict(device, transform, model, image_path, config['Classes'].replace(" ", "").split(','))
        print(f"The {config['ClassifiedObject']} is {prediction}.")
    
    else:
        print_usage()
        sys.exit(1)


if __name__ == '__main__':
    main()

