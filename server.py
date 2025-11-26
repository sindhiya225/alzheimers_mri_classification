import flwr as fl
import torch
from model import AlzheimerCNN
from flwr.common import parameters_to_ndarrays
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from utils import DATASET_DIR

# Custom strategy to save the global model
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(on_fit_config_fn=lambda rnd: {"round": rnd})

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, _ = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            aggregated_weights = parameters_to_ndarrays(aggregated_parameters)

            model = AlzheimerCNN()
            params_dict = zip(model.state_dict().keys(), aggregated_weights)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)

            # ✅ Save global model
            torch.save(model.state_dict(), "global_model.pth")
            print(f"✅ Saved global model at round {server_round}")

            #  Evaluate global model on centralized test set
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor()
            ])
            test_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"), transform=transform)
            testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for data, target in testloader:
                    data, target = data.to("cpu"), target.to("cpu")
                    outputs = model(data)
                    _, predicted = torch.max(outputs, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            acc = 100 * correct / total
            print(f" Global Test Accuracy after round {server_round}: {acc:.2f}%")

        return aggregated_parameters, {}


if __name__ == "__main__":
    strategy = SaveModelStrategy()
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )
