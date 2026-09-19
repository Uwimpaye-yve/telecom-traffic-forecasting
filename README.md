# Mobile Network Traffic Forecasting: Sequential Model Benchmarking

An empirical research investigation comparing statistical autoregressive, recurrent, and dilated convolutional architectures for one-step-ahead cellular Internet traffic forecasting using the Telecom Italia Milan Big Data dataset.

---

## Project Structure

telecom-traffic-forecasting/
|-- data/
|   |-- raw/                 # Ignored by git: 62 raw daily files
|   +-- processed/           # Filtered and compressed Parquet files
|-- notebooks/               # Interactive exploration
|-- outputs/
|   |-- figures/             # Spatial, temporal diagnostics & 9 superposed forecast plots
|   +-- tables/              # Empirical performance CSVs
|-- src/
|   |-- data/
|   |   |-- download_data.py # Automated Harvard Dataverse API fetcher
|   |   +-- process_data.py  # Memory-optimized chunked parser and Parquet converter
|   |-- eda/
|   |   |-- exploratory_analysis.py # Spatial distribution, ADF test, ACF/PACF, STL
|   |   +-- failure_analysis.py     # San Siro Stadium (Square 4556) anomaly stress test
|   +-- models/
|       +-- train_evaluate.py       # Sliding window loader, SARIMAX, LSTM, and TCN pipelines
|-- .gitignore
|-- requirements.txt
+-- README.md

---

## Hardware and Compute Environment

All benchmarks were recorded under the following configuration:
* Operating System: Windows 10/11 64-bit
* Runtime: Python 3.13
* Compute Device: CPU (torch.device('cpu'))
* Profiling Method: time.perf_counter() execution wall time

---

## Setup and Reproduction

### 1. Environment Setup
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Data Processing & Memory Optimization
Run chunked ingestion, type downcasting (uint16, float32), and Parquet compression:
```bash
python src/data/process_data.py
```
* Memory Reduction Achieved: 78.12% (12.21 MB down to 2.67 MB per 200k rows).
* Identified Top 3 Areas: Square ID 5161, 5059, and 5259.

### 3. Exploratory Data Analysis & Diagnostics
Generate the spatial distribution, 5-area comparison, ADF stationarity test, and ACF/PACF plots:
```bash
python src/eda/exploratory_analysis.py
```

### 4. Model Training & Comparative Evaluation
Train SARIMAX, Stacked LSTM, and Dilated TCN across all three target zones for the evaluation week (December 16-22, 2013):
```bash
python src/models/train_evaluate.py
```

### 5. Failure Case Analysis
Simulate and generate the exogenous event breakdown on Square ID 4556:
```bash
python src/eda/failure_analysis.py
```

---

## Empirical Benchmark Results

Evaluated over the test week (December 16-22, 2013, N=1008 intervals):

| Square ID | Model Architecture | MAE | RMSE | MAPE (%) | Train Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **5161 (Rank 1)** | SARIMAX(1,0,1)(1,0,0)144 | **93.06** | **135.01** | **9.29%** | 81.01 |
| | Stacked LSTM | 100.96 | 145.72 | 13.30% | 150.74 |
| | Dilated TCN | 133.11 | 185.83 | 18.26% | **79.50** |
| **5059 (Rank 2)** | SARIMAX(1,0,1)(1,0,0)144 | 81.45 | 114.35 | 7.96% | 116.40 |
| | Stacked LSTM | 84.68 | 113.84 | 10.75% | 107.82 |
| | Dilated TCN | **71.13** | **101.55** | **7.34%** | **41.52** |
| **5259 (Rank 3)** | SARIMAX(1,0,1)(1,0,0)144 | 76.02 | 109.60 | 8.15% | 130.67 |
| | Stacked LSTM | 71.18 | 102.46 | 7.68% | 126.46 |
| | Dilated TCN | **69.59** | **99.34** | **7.69%** | **41.90** |
