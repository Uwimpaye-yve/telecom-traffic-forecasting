import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

PROCESSED_DIR = os.path.join("data", "processed")
FIGURES_DIR = os.path.join("outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def evaluate_failure_case():
    print("Simulating event failure dynamics on Square ID 4556...")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "target_squares_traffic.parquet"))
    sub = df[df["square_id"] == 4556].drop_duplicates(subset=["datetime"]).sort_values("datetime")
    sub = sub.set_index("datetime")["internet"].asfreq("10min").ffill().reset_index()

    window_size = 144
    train_raw = sub[sub["datetime"] < "2013-12-16"]["internet"].values.reshape(-1, 1)
    test_mask = (sub["datetime"] >= "2013-12-16 00:00:00") & (sub["datetime"] <= "2013-12-22 23:50:00")
    
    test_start_idx = np.where(test_mask)[0][0]
    test_end_idx = np.where(test_mask)[0][-1]
    test_slice = sub.iloc[test_start_idx - window_size : test_end_idx + 1]["internet"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_slice)

    # Prepare sliding windows
    X_train, y_train = [], []
    for i in range(len(train_scaled) - window_size):
        X_train.append(train_scaled[i:i + window_size])
        y_train.append(train_scaled[i + window_size])
    X_train, y_train = np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32)

    X_test, y_test = [], []
    for i in range(len(test_scaled) - window_size):
        X_test.append(test_scaled[i:i + window_size])
        y_test.append(test_scaled[i + window_size])
    X_test = np.array(X_test, dtype=np.float32)

    actual = scaler.inverse_transform(np.array(y_test)).flatten()

    model = LSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=64, shuffle=True
    )
    
    model.train()
    for _ in range(8):
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(X_test)).numpy()
    pred = scaler.inverse_transform(pred_scaled).flatten()

    # Plot failure case
    dt_index = sub[test_mask]["datetime"]
    plt.figure(figsize=(14, 4.5))
    plt.plot(dt_index, actual, label="Actual Observed (Square 4556)", color="black", linewidth=1.2)
    plt.plot(dt_index, pred, label="LSTM Forecast (Missed Surge)", color="#d62728", linestyle="--", linewidth=1.2)
    plt.title("Failure Case: Inability of Univariate Sequence Models to Predict Exogenous Bursts (Square 4556)", fontsize=11)
    plt.xlabel("Datetime")
    plt.ylabel("Internet Traffic")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.4)

    save_path = os.path.join(FIGURES_DIR, "fig5_failure_case_4556.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved failure case visualization: {save_path}")

if __name__ == "__main__":
    evaluate_failure_case()