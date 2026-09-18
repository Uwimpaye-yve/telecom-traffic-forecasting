import os
import requests

DATASET_DOI = "doi:10.7910/DVN/EGZHFV"
API_BASE = "https://dataverse.harvard.edu/api"

print("Fetching dataset metadata from Harvard Dataverse...")
url = f"{API_BASE}/datasets/:persistentId/?persistentId={DATASET_DOI}"
response = requests.get(url).json()
files = response["data"]["latestVersion"]["files"]

# Download destination
target_dir = os.path.join("data", "raw")
os.makedirs(target_dir, exist_ok=True)

print(f"Discovered {len(files)} files. Starting streaming download...")

for idx, item in enumerate(files, 1):
    filename = item["dataFile"]["filename"]
    file_id = item["dataFile"]["id"]
    save_path = os.path.join(target_dir, filename)

    if os.path.exists(save_path):
        print(f"[{idx}/{len(files)}] Skipping {filename} (already exists)")
        continue

    print(f"[{idx}/{len(files)}] Downloading {filename}...")
    download_url = f"{API_BASE}/access/datafile/{file_id}"
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

print("All files downloaded successfully.")