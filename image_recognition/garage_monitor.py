import os
import time
import configparser
import torch
import torch.nn as nn
import requests
from PIL import Image
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

torch.set_float32_matmul_precision('high')

IMAGE_PATH = "/mnt/ramdisk/garage.jpg"
CHECK_INTERVAL = 10  # seconds

def predict(device, transform, model, image_path, classes):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return classes[predicted.item()]

def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = MobileNet_V3_Small_Weights.DEFAULT
    transform = weights.transforms()

    model = mobilenet_v3_small(weights=weights)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )

    model_path = f'./{model_name}.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()

    return device, transform, model

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['DEFAULT']

def main():
    config = load_config()
    device, transform, model = load_model(config['ModelName'])
    classes = config['Classes'].replace(" ", "").split(',')

    while True:
        if os.path.exists(IMAGE_PATH):
            try:
                result = predict(device, transform, model, IMAGE_PATH, classes)
                requests.post("http://0.0.0.0:5000/status", json={"status": result.capitalize()})
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Prediction: {result}")
            except Exception as e:
                print(f"[ERROR] Failed to process image: {e}")
        else:
            print(f"[WARN] Image not found at {IMAGE_PATH}")

        time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    main()

