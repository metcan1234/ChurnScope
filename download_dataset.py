#!/usr/bin/env python3
"""Download the Telco Customer Churn dataset from Kaggle or direct URL."""
import os
import sys
import shutil
import zipfile
import urllib.request
import pandas as pd
from io import BytesIO

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def download_from_kaggle():
    """Try to download using kagglehub."""
    try:
        import kagglehub
        path = kagglehub.dataset_download("blastchar/telco-customer-churn")
        print(f"Downloaded to: {path}")
        # Find the CSV file in the downloaded path
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith(".csv"):
                    src = os.path.join(root, f)
                    dst = os.path.join(DATA_DIR, CSV_FILENAME)
                    shutil.copy2(src, dst)
                    print(f"Copied {f} to {dst}")
                    return True
    except Exception as e:
        print(f"kagglehub failed: {e}")
    return False

def download_direct():
    """Download from a direct URL (GitHub mirror of the dataset)."""
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    try:
        print(f"Downloading from {url}...")
        with urllib.request.urlopen(url) as response:
            df = pd.read_csv(response)
            dst = os.path.join(DATA_DIR, CSV_FILENAME)
            df.to_csv(dst, index=False)
            print(f"Saved to {dst} ({df.shape[0]} rows, {df.shape[1]} cols)")
            return True
    except Exception as e:
        print(f"Direct download failed: {e}")
    return False

def check_existing():
    """Check if dataset already exists."""
    dst = os.path.join(DATA_DIR, CSV_FILENAME)
    if os.path.exists(dst):
        df = pd.read_csv(dst)
        print(f"Dataset already exists: {dst} ({df.shape[0]} rows, {df.shape[1]} cols)")
        return True
    return False

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if check_existing():
        return
    
    print("Downloading Telco Customer Churn dataset...")
    
    if download_from_kaggle():
        return
    
    if download_direct():
        return
    
    print("ERROR: Could not download dataset. Please download manually from:")
    print("  https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
    sys.exit(1)

if __name__ == "__main__":
    main()