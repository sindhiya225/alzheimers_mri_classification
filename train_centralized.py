import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import AlzheimerCNN
import os

# 🔹 Absolute dataset path
DATASET_DIR = "C:/Users/Nitharshna A/OneDrive/Desktop/alzheimer_fl/data/Alzheimer_s Dataset"

train_dir = os.path.join(DATASET_DIR, "train")
test_dir = os.path.join(DATASET_DIR, "test")
os.makedirs("metrics", exist_ok=True)

# 🔹 Data transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 🔹 Datasets & loaders
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 🔹 Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlzheimerCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 🔹 Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for data, target in trainloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(trainloader):.4f}")

    # 🔹 Evaluate on test set
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
    print(f"Epoch {epoch+1} Accuracy: {accuracy:.2f}%")

    # 🔹 Save accuracy metrics (like in FL)
    with open("metrics/centralized_metrics.csv", "a") as f:
        f.write(f"{epoch+1},{accuracy:.2f}\n")

# 🔹 Save final trained model
torch.save(model.state_dict(), "centralized_model.pth")
print("✅ Saved centralized model as centralized_model.pth")
