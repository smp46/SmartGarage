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

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
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

# Training Function
def train(model, train_loader, criterion, optimizer, num_epochs=10):
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

# Example Data Setup (Replace with your actual base path)
base_path = r"F:\UserData\OneDrive\Documents\GarageDoor\GaragePhotos"  # Adjust this path to where your 'open' and 'closed' folders are

# DataLoader
dataset = GarageDoorDataset(base_path, transform=transform)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Start Training
train(model, train_loader, criterion, optimizer, num_epochs=10)
