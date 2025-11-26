import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import AlzheimerCNN
import os

# 🔹 Paths to both models
MODELS = {
    "Centralized": "centralized_model.pth",
    "Federated": "global_model.pth"
}

# 🔹 Dataset path (adjust to your absolute path)
DATASET_DIR = r"D:\PSG\SEMESTER 5\Mentor Project\cnn_alz\data\Alzheimer_s Dataset"
test_dir = os.path.join(DATASET_DIR, "test")

# 🔹 Transform and loader
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model_path, model_name):
    model = AlzheimerCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f"✅ {model_name} Model: {model_path}")
    print(f"✅ Evaluated on {total} images")
    print(f"🎯 Accuracy: {accuracy:.2f}%\n")

# 🔹 Evaluate both models
for name, path in MODELS.items():
    if os.path.exists(path):
        evaluate_model(path, name)
    else:
        print(f"⚠️ {name} model not found: {path}")
