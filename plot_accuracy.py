import pandas as pd
import matplotlib.pyplot as plt
import os

# ---- File Paths ----
CENTRALIZED_METRICS = "metrics/centralized_metrics.csv"
FEDERATED_GLOBAL_METRICS = "metrics/federated_global_metrics.csv"
CLIENT_METRICS_PATTERN = "metrics/client{}_metrics.csv"

print("\n📊 ===== ACCURACY SUMMARY =====")

# ---- Centralized Test Accuracy ----
if os.path.exists(CENTRALIZED_METRICS):
    df_central = pd.read_csv(CENTRALIZED_METRICS, header=None, names=["epoch", "accuracy"])
    best_epoch = df_central["accuracy"].idxmax()
    best_acc = df_central.loc[best_epoch, "accuracy"]
    print(f"🟢 Centralized Model → Highest Test Accuracy: {best_acc:.2f}% (at Epoch {df_central.loc[best_epoch, 'epoch']})")

    plt.figure(figsize=(7,5))
    plt.plot(df_central["epoch"], df_central["accuracy"], 'g-o', label="Centralized Test Accuracy")
    plt.title("Centralized Model - Test Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Centralized metrics file not found!")

# ---- Federated Global Test Accuracy ----
if os.path.exists(FEDERATED_GLOBAL_METRICS):
    df_global = pd.read_csv(FEDERATED_GLOBAL_METRICS, header=None, names=["round", "accuracy"])
    best_round = df_global["accuracy"].idxmax()
    best_acc = df_global.loc[best_round, "accuracy"]
    print(f"🔴 Federated Global Model → Highest Test Accuracy: {best_acc:.2f}% (at Round {df_global.loc[best_round, 'round']})")

    plt.figure(figsize=(7,5))
    plt.plot(df_global["round"], df_global["accuracy"], 'r-s', label="Federated Global Test Accuracy")
    plt.title("Federated Global Model - Test Accuracy per Round")
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Federated global metrics file not found!")

# ---- Client Local Accuracies (Train Accuracy per Round) ----
for i in range(3): # adjust if you have more clients
    client_file = CLIENT_METRICS_PATTERN.format(i)
    if os.path.exists(client_file):
        df_client = pd.read_csv(client_file, header=None, names=["round", "accuracy"])
        best_round = df_client["accuracy"].idxmax()
        best_acc = df_client.loc[best_round, "accuracy"]
        print(f"🧠 Client {i} → Highest Local (Train) Accuracy: {best_acc:.2f}% (at Round {df_client.loc[best_round, 'round']})")

        plt.figure(figsize=(7,5))
        plt.plot(df_client["round"], df_client["accuracy"], '--o', label=f"Client {i} Train Accuracy")
        plt.title(f"Client {i} - Local Train Accuracy per Round")
        plt.xlabel("Round")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print(f"⚠️ Metrics for Client {i} not found!")

print("\n✅ Done! All graphs displayed and accuracies summarized.")