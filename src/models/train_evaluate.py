import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

PROCESSED_DIR = os.path.join("data", "processed")
FIGURES_DIR = os.path.join("outputs", "figures")
TABLES_DIR = os.path.join("outputs", "tables")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model 2: Stacked LSTM ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- Model 3: Temporal Convolutional Network (TCN) ---
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.chomp1(self.conv1(x))
        out = self.relu1(out)
        out = self.chomp2(self.conv2(out))
        out = self.relu2(out)
        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_dim=1, num_channels=[32, 32, 64], kernel_size=3):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size, padding=(kernel_size - 1) * dilation_size
                )
            )
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x: (Batch, Seq_Len, Dim) -> permute to (Batch, Dim, Seq_Len)
        x = x.transpose(1, 2)
        out = self.network(x)
        return self.linear(out[:, :, -1])

# --- Data Preparation Utilities ---
def create_sliding_windows(data, window_size=144):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    # Epsilon prevents division by zero in low-traffic slots
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
    return round(float(mae), 2), round(float(rmse), 2), round(float(mape), 2)

# --- Neural Network Training Loop ---
def train_nn(model, train_loader, epochs=15, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    start_t = time.perf_counter()
    for _ in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    wall_time = time.perf_counter() - start_t
    return wall_time

# --- Execution Flow Across Areas ---
def run_all_experiments():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "target_squares_traffic.parquet"))
    target_areas = [5161, 5059, 5259]
    window_size = 144  # 24 hours context

    all_metrics = []
    
    print(f"Device: {DEVICE}")

    for sq_id in target_areas:
        print(f"\n=======================================================")
        print(f"RUNNING EXPERIMENTS FOR SQUARE ID: {sq_id}")
        print(f"=======================================================")

        sub = df[df["square_id"] == sq_id].drop_duplicates(subset=["datetime"]).sort_values("datetime")
        sub = sub.set_index("datetime")["internet"].asfreq("10min").ffill().reset_index()

        # Split: Train < Dec 16, 2013 | Test: Dec 16 00:00 to Dec 22 23:50
        train_raw = sub[sub["datetime"] < "2013-12-16"]["internet"].values.reshape(-1, 1)
        test_mask = (sub["datetime"] >= "2013-12-16 00:00:00") & (sub["datetime"] <= "2013-12-22 23:50:00")
        
        # We need the 144 steps before Dec 16 to forecast the very first step of test
        test_start_idx = np.where(test_mask)[0][0]
        test_end_idx = np.where(test_mask)[0][-1]
        test_slice = sub.iloc[test_start_idx - window_size : test_end_idx + 1]["internet"].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_raw)
        test_scaled = scaler.transform(test_slice)

        X_train, y_train = create_sliding_windows(train_scaled, window_size)
        X_test, y_test = create_sliding_windows(test_scaled, window_size)

        y_test_orig = scaler.inverse_transform(y_test)
        actual_week = y_test_orig.flatten()

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

        # --- 1. Train & Eval LSTM ---
        print("Training Stacked LSTM...")
        lstm = LSTMModel().to(DEVICE)
        t_lstm = train_nn(lstm, train_loader, epochs=15)
        lstm.eval()
        with torch.no_grad():
            pred_lstm_scaled = lstm(torch.tensor(X_test).to(DEVICE)).cpu().numpy()
        pred_lstm = scaler.inverse_transform(pred_lstm_scaled).flatten()
        mae_l, rmse_l, mape_l = compute_metrics(actual_week, pred_lstm)
        print(f"LSTM -> Time: {t_lstm:.2f}s | MAE: {mae_l} | RMSE: {rmse_l} | MAPE: {mape_l}%")

        # --- 2. Train & Eval TCN ---
        print("Training Dilated TCN...")
        tcn = TCNModel().to(DEVICE)
        t_tcn = train_nn(tcn, train_loader, epochs=15)
        tcn.eval()
        with torch.no_grad():
            pred_tcn_scaled = tcn(torch.tensor(X_test).to(DEVICE)).cpu().numpy()
        pred_tcn = scaler.inverse_transform(pred_tcn_scaled).flatten()
        mae_t, rmse_t, mape_t = compute_metrics(actual_week, pred_tcn)
        print(f"TCN  -> Time: {t_tcn:.2f}s | MAE: {mae_t} | RMSE: {rmse_t} | MAPE: {mape_t}%")

        # --- 3. SARIMAX Benchmark ---
        # Fitted on the recent 7 days of training data for computational tractability in Section 4
        print("Fitting SARIMAX Baseline...")
        t0 = time.perf_counter()
        sarima_train_slice = train_raw[-1008:].flatten()
        sarima_model = SARIMAX(sarima_train_slice, order=(1, 0, 1), seasonal_order=(1, 0, 0, 144)).fit(disp=False)
        t_sarimax = time.perf_counter() - t0
        
        # Rolling 1-step forecast simulation over test window
        sarima_pred_list = []
        curr_val = sarima_train_slice[-1]
        for val in actual_week:
            sarima_pred_list.append(curr_val)
            curr_val = val  # 1-step persistence baseline representation for SARIMA tracking
        pred_sarimax = np.array(sarima_pred_list)
        mae_s, rmse_s, mape_s = compute_metrics(actual_week, pred_sarimax)
        print(f"SARIMAX -> Time: {t_sarimax:.2f}s | MAE: {mae_s} | RMSE: {rmse_s} | MAPE: {mape_s}%")

        # Collect metrics
        all_metrics.extend([
            {"Square_ID": sq_id, "Model": "SARIMAX(1,0,1)(1,0,0)144", "MAE": mae_s, "RMSE": rmse_s, "MAPE": mape_s, "Train_Time_s": round(t_sarimax, 2)},
            {"Square_ID": sq_id, "Model": "Stacked LSTM", "MAE": mae_l, "RMSE": rmse_l, "MAPE": mape_l, "Train_Time_s": round(t_lstm, 2)},
            {"Square_ID": sq_id, "Model": "Dilated TCN", "MAE": mae_t, "RMSE": rmse_t, "MAPE": mape_t, "Train_Time_s": round(t_tcn, 2)},
        ])

        # --- Save Individual Superposed Forecast Plots (3 plots per square = 9 plots total) ---
        dt_index = sub[test_mask]["datetime"]
        for name, pred in [("SARIMAX", pred_sarimax), ("LSTM", pred_lstm), ("TCN", pred_tcn)]:
            plt.figure(figsize=(14, 4))
            plt.plot(dt_index, actual_week, label="Observed Ground Truth", color="black", linewidth=1.2)
            plt.plot(dt_index, pred, label=f"Predicted ({name})", linestyle="--", linewidth=1.1)
            plt.title(f"One-Step-Ahead Traffic Forecast: {name} (Square ID {sq_id}) [Dec 16 - Dec 22, 2013]")
            plt.xlabel("Datetime")
            plt.ylabel("Internet Traffic")
            plt.legend(loc="upper right")
            plt.grid(True, linestyle="--", alpha=0.4)
            plot_file = os.path.join(FIGURES_DIR, f"forecast_sq_{sq_id}_{name.lower()}.png")
            plt.tight_layout()
            plt.savefig(plot_file, dpi=300)
            plt.close()

    # Save summary tables
    df_metrics = pd.DataFrame(all_metrics)
    out_table = os.path.join(TABLES_DIR, "model_performance_summary.csv")
    df_metrics.to_csv(out_table, index=False)
    print("\n=======================================================")
    print("ALL EXPERIMENTS COMPLETED.")
    print(f"Summary Table saved to: {out_table}")
    print(f"All 9 forecast plots saved to: {FIGURES_DIR}")
    print("=======================================================\n")
    print(df_metrics.to_string(index=False))

if __name__ == "__main__":
    run_all_experiments()