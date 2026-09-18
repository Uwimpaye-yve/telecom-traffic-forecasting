import glob
import os
import time
import numpy as np
import pandas as pd
import psutil

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

COL_NAMES = [
    "square_id", "time_interval", "country_code",
    "sms_in", "sms_out", "call_in", "call_out", "internet"
]

DTYPES = {
    "square_id": "uint16",
    "time_interval": "int64",
    "internet": "float32"
}

def profile_memory_usage(sample_file):
    print("\n--- Memory Profiling Benchmark (Single Day File) ---")
    
    # 1. Naive load (default pandas behavior: int64, float64, all 8 columns)
    t0 = time.perf_counter()
    df_naive = pd.read_csv(sample_file, sep="\t", header=None, names=COL_NAMES, nrows=200000)
    naive_mem = df_naive.memory_usage(deep=True).sum() / (1024 ** 2)
    naive_time = time.perf_counter() - t0
    
    # 2. Optimized load (column projection + downcasting)
    t0 = time.perf_counter()
    df_opt = pd.read_csv(
        sample_file,
        sep="\t",
        header=None,
        names=COL_NAMES,
        usecols=["square_id", "time_interval", "internet"],
        dtype=DTYPES,
        nrows=200000
    )
    opt_mem = df_opt.memory_usage(deep=True).sum() / (1024 ** 2)
    opt_time = time.perf_counter() - t0
    
    reduction = (1 - (opt_mem / naive_mem)) * 100
    print(f"Naive Ingestion (200k rows):     {naive_mem:.2f} MB ({naive_time:.2f}s)")
    print(f"Optimized Ingestion (200k rows): {opt_mem:.2f} MB ({opt_time:.2f}s)")
    print(f"Memory Reduction Achieved:       {reduction:.2f}%\n")

def run_pipeline():
    raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "sms-call-internet-mi-*.txt")))
    if not raw_files:
        print("No raw text files found in data/raw/")
        return

    profile_memory_usage(raw_files[0])

    print("Phase 1: Aggregating total traffic per grid square across all 62 days...")
    total_grid_traffic = np.zeros(10001, dtype=np.float64)
    start_total = time.perf_counter()

    for idx, fpath in enumerate(raw_files, 1):
        chunks = pd.read_csv(
            fpath,
            sep="\t",
            header=None,
            names=COL_NAMES,
            usecols=["square_id", "internet"],
            dtype={"square_id": "uint16", "internet": "float32"},
            chunksize=500_000
        )
        for chunk in chunks:
            chunk = chunk.dropna(subset=["internet"])
            grouped = chunk.groupby("square_id")["internet"].sum()
            total_grid_traffic[grouped.index.values] += grouped.values
        
        if idx % 10 == 0 or idx == len(raw_files):
            print(f"Processed [{idx}/{len(raw_files)}] daily files...")

    top_indices = np.argsort(total_grid_traffic)[::-1][:3]
    print("\n=======================================================")
    print("TOP 3 GEOGRAPHICAL AREAS IDENTIFIED OVER 2 MONTHS:")
    for rank, sq_id in enumerate(top_indices, 1):
        print(f"Rank {rank}: Square ID {sq_id} (Total Internet: {total_grid_traffic[sq_id]:,.2f})")
    print("=======================================================\n")

    target_squares = set(top_indices.tolist() + [4159, 4556])
    print(f"Phase 2: Extracting complete time series for target cells: {sorted(list(target_squares))}...")

    records = []
    for idx, fpath in enumerate(raw_files, 1):
        chunks = pd.read_csv(
            fpath,
            sep="\t",
            header=None,
            names=COL_NAMES,
            usecols=["square_id", "time_interval", "internet"],
            dtype=DTYPES,
            chunksize=500_000
        )
        for chunk in chunks:
            chunk = chunk.dropna(subset=["internet"])
            filtered = chunk[chunk["square_id"].isin(target_squares)]
            if not filtered.empty:
                agg = filtered.groupby(["square_id", "time_interval"], as_index=False)["internet"].sum()
                records.append(agg)

    df_targets = pd.concat(records, ignore_index=True)
    df_targets = df_targets.groupby(["square_id", "time_interval"], as_index=False)["internet"].sum()
    
    df_targets["datetime"] = pd.to_datetime(df_targets["time_interval"], unit="ms")
    df_targets = df_targets.sort_values(["square_id", "datetime"]).reset_index(drop=True)

    out_parquet = os.path.join(PROCESSED_DIR, "target_squares_traffic.parquet")
    df_targets.to_parquet(out_parquet, index=False, engine="pyarrow")
    
    df_spatial = pd.DataFrame({
        "square_id": np.arange(1, 10001),
        "total_internet": total_grid_traffic[1:]
    })
    df_spatial.to_parquet(os.path.join(PROCESSED_DIR, "spatial_grid_totals.parquet"), index=False)

    print(f"\nPipeline Complete in {time.perf_counter() - start_total:.1f}s.")
    print(f"Saved targets time series to: {out_parquet}")
    print(f"Saved spatial grid summary to: {os.path.join(PROCESSED_DIR, 'spatial_grid_totals.parquet')}")

if __name__ == "__main__":
    run_pipeline()