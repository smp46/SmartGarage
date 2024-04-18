import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Define the Dataset Class
class GarageDoorDataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.transform = transform
        # Load images and labels
        self.data = []
        for label, folder in enumerate(['closed', 'open']):
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

# Training Function
def train(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def train_main():
    # Image Transformations
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Pre-trained MobileNetV2 and Modify It
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, 2)  # Change for binary classification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Example Data Setup (Replace with your actual base path)
    base_path = r"F:\UserData\OneDrive\Documents\GarageDoor\GaragePhotos"  # Adjust this path to where your 'open' and 'closed' folders are

    # DataLoader
    dataset = GarageDoorDataset(base_path, transform=transform)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Start Training
    train(model, train_loader, criterion, optimizer, device, num_epochs=5)
    
    # Save trained model and dataset
    model_path = './smartGarage.pth'
    torch.save(model.state_dict(), model_path)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    return device, transform, model

def validater(device=None, transform=None, model=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if transform is None:
        transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    if model is None:
        model = models.mobilenet_v2()  # No need to load pretrained weights now
        model.classifier[1] = nn.Linear(model.last_channel, 2)
        model_path = './smartGarage.pth'
        model.load_state_dict(torch.load(model_path))
        model.to(device)
    
    # Assuming you have validation data prepared
    val_base_path = r"F:\UserData\OneDrive\Documents\GarageDoor\TestPhotos"
    val_labels = [0, 1]  # 0 for closed, 1 for open

    val_dataset = GarageDoorDataset(val_base_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Function to evaluate the model
    def evaluate(model, val_loader):
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
        print(f'Accuracy: {accuracy:.2f}%')
        return accuracy

    # Evaluate the model
    return evaluate(model, val_loader)
    
def predict(image_path, device=None, transform=None, model=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if transform is None:
        transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    if model is None:
        model = models.mobilenet_v2()  # No need to load pretrained weights now
        model.classifier[1] = nn.Linear(model.last_channel, 2)
        model_path = './smartGarage.pth'
        model.load_state_dict(torch.load(model_path))
        model.to(device)
    
    
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return 'Open' if predicted.item() == 1 else 'Closed'

def main():
    # Train the model
    #device, transform, model = train_main()
    #total_accuracy = validater(device, transform, model)
    
    #total_accuracy = validater()
    
    #print(f'Total Accuracy: {total_accuracy:.2f}%')

    # # Validate the model
    # validater()

    # Predict on new image
    image_path = r"F:\UserData\OneDrive\Documents\GarageDoor\TestPhotos\lighton.jpg"  # Add your new image path
    prediction = predict(image_path)
    print(f'The garage door is {prediction}')

if __name__ == '__main__':
    main()

