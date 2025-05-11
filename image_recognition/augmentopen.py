import os
from PIL import Image
import torchvision.transforms as T
from glob import glob
from tqdm import tqdm

input_dir = '/train/training_imgs_sorted/open'
output_dir = '/train/training_imgs_sorted/open_augmented'
os.makedirs(output_dir, exist_ok=True)

transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    T.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    T.RandomPerspective(distortion_scale=0.2, p=0.5),
    T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
])

image_paths = glob(f"{input_dir}/*.jpg")

num_augmentations = 100  # per image

for img_path in tqdm(image_paths):
    img = Image.open(img_path).convert("RGB")
    base_name = os.path.basename(img_path).split('.')[0]
    
    for i in range(num_augmentations):
        aug_img = transform(img)
        aug_img.save(os.path.join(output_dir, f"{base_name}_aug{i}.jpg"))

