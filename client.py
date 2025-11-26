import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from model import AlzheimerCNN
from utils import get_dataloader, split_dataset_into_clients, DATASET_DIR
import sys
import os

client_id = int(sys.argv[1])

# Split dataset for clients (absolute path)
split_dataset_into_clients(
    source_dir=os.path.join(DATASET_DIR, "train"),
    output_dir="data/split_data",
    num_clients=3
)

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader):
        self.model = model
        self.trainloader = trainloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        # ✅ Each client evaluates on its own local split
        testloader = get_dataloader(f"data/split_data/client{client_id}")

        correct, total = 0, 0
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
        print(f"[Client {client_id}] Local Test Accuracy: {accuracy:.2f}%")

        os.makedirs("metrics", exist_ok=True)
        with open(f"metrics/client{client_id}_metrics.csv", "a") as f:
            f.write(f"{config.get('round', 0)},{accuracy:.2f}\n")

        return float(1 - accuracy / 100), total, {"accuracy": accuracy}


model = AlzheimerCNN()
trainloader = get_dataloader(f"data/split_data/client{client_id}")
fl.client.start_client(
    server_address="localhost:8080",
    client=FLClient(model, trainloader).to_client()
)
