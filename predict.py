import torch
from torchvision import transforms
from PIL import Image
from model import AlzheimerCNN

# Load trained model
model = AlzheimerCNN()
model.load_state_dict(torch.load("centralized_model.pth", map_location="cpu"))
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

img = Image.open("sample_mri.jpg").convert("RGB")
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)

classes = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

print(f"Predicted class: {classes[predicted.item()]}")


#model.load_state_dict(torch.load("centralized_model.pth", map_location="cpu"))
#model.load_state_dict(torch.load("global_model.pth", map_location="cpu"))  # or "centralized_model.pth"
