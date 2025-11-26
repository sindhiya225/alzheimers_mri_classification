import os
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Absolute dataset path
DATASET_DIR = "C:/Users/Nitharshna A/OneDrive/Desktop/alzheimer_fl/data/Alzheimer_s Dataset"

def split_dataset_into_clients(source_dir=None, output_dir="data/split_data", num_clients=3):
    if source_dir is None:
        source_dir = os.path.join(DATASET_DIR, "train")

    classes = os.listdir(source_dir)
    for client_id in range(num_clients):
        for cls in classes:
            os.makedirs(os.path.join(output_dir, f"client{client_id}", cls), exist_ok=True)

    for cls in classes:
        class_dir = os.path.join(source_dir, cls)
        images = os.listdir(class_dir)
        split_size = len(images) // num_clients
        for client_id in range(num_clients):
            start = client_id * split_size
            end = (client_id + 1) * split_size if client_id != num_clients - 1 else len(images)
            for img_name in images[start:end]:
                shutil.copy(
                    os.path.join(class_dir, img_name),
                    os.path.join(output_dir, f"client{client_id}", cls)
                )

def get_dataloader(data_dir):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=16, shuffle=True)
