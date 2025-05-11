import sys
import os
import configparser
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import ConvNextV2ForImageClassification, AutoImageProcessor
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# Define the Dataset Class
class GarageDoorDataset(Dataset):
    def __init__(self, base_path, image_processor, classes):
        class0, class1 = classes
        self.image_processor = image_processor
        # Load images and labels
        self.data = []
        for label, folder in enumerate([class0, class1]):
            folder_path = os.path.join(base_path, folder)
            images = glob.glob(f'{folder_path}/*.jpg')  # Assuming images are in .jpg format
            for image in images:
                self.data.append((image, label))
        
        print(f"Total images loaded: {len(self.data)}")  # Debugging line

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        # Process image using Hugging Face's AutoImageProcessor
        pixel_values = self.image_processor(images=image, return_tensors='pt').pixel_values.squeeze(0)
        return pixel_values, label


def train(model, train_loader, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{num_epochs}"
        )

        for batch_idx, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update progress bar and loss
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} completed with Average Loss: {epoch_loss / len(train_loader):.4f}")

# Model Loading Function
def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "error")
    # Load ConvNeXt V2 model and processor
    model = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-base-22k-224").to(device)
    image_processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-base-22k-224", use_fast=True)
    return model, image_processor, device

# Main Training Function
def train_main(classes, num_epochs, model_name, images_path):
    model, image_processor, device = load_model(model_name)

    # Prepare dataset and dataloader
    dataset = GarageDoorDataset(images_path, image_processor, classes)
    train_loader = DataLoader(dataset, batch_size=8, num_workers=8, shuffle=True)

    # Set up optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Start training
    train(model, train_loader, criterion, optimizer, device, num_epochs)

    # Save trained model
    model_path = f'./{model_name}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def validater(device, image_processor, model, val_base_path, classes):
    val_dataset = GarageDoorDataset(val_base_path, image_processor, classes)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    return evaluate(model, val_loader, device)

# Function to evaluate the model
def evaluate(model, val_loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def predict(device, image_processor, model, image_path, classes):
    class0, class1 = classes
    model.eval()
    image = Image.open(image_path).convert('RGB')
    pixel_values = image_processor(images=image, return_tensors='pt').pixel_values.to(device)
    with torch.no_grad():
        outputs = model(pixel_values).logits
        _, predicted = torch.max(outputs, 1)
    return class0 if predicted.item() == 0 else class1

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
        model, image_processor, device = load_model(config['ModelName'])
        classes = config['Classes'].split(',')
        accuracy = validater(device, image_processor, model, val_path, classes)
        print(f'The model accuracy is {accuracy}%')

    elif action == 'predict':
        image_path = input("Enter the path to the image for prediction: ")
        config = load_config()
        model, image_processor, device = load_model(config['ModelName'])
        prediction = predict(device, image_processor, model, image_path, config['Classes'].replace(" ", "").split(','))
        print(f"The {config['ClassifiedObject']} is {prediction}.")
    else:
        print_usage()
        sys.exit(1)

if __name__ == '__main__':
    main()

