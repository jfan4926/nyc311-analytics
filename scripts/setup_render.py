import os
import subprocess
import urllib.parse
import pandas as pd

DB_DIR  = "data/processed"
RAW_DIR = "data/raw"

def download_data():
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(f"{RAW_DIR}/nyc311.parquet"):
        print("Data already exists, skipping download")
        return
    print("Downloading NYC 311 data...")
    params = urllib.parse.urlencode({
        "$limit": 100000,
        "$where": "created_date >= '2022-01-01'",
        "$order": "created_date DESC"
    })
    url = f"https://data.cityofnewyork.us/resource/erm2-nwe9.csv?{params}"
    df  = pd.read_csv(url)
    df.to_parquet(f"{RAW_DIR}/nyc311.parquet", index=False)
    print(f"Downloaded {len(df):,} rows")

def run_dbt():
    os.makedirs(DB_DIR, exist_ok=True)
    print("Running dbt...")
    result = subprocess.run(
        ["python", "-m", "dbt", "run", "--profiles-dir", "."],
        cwd="dbt_project/nyc311",
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("dbt run failed")

def train_model():
    if os.path.exists("ml_pipeline/models/encoders.pkl"):
        print("Model already exists, skipping training")
        return
    print("Training model...")
    subprocess.run(["python", "ml_pipeline/train.py"], check=True)

def build_rag():
    if os.path.exists("api/llm/chroma_store"):
        print("RAG store already exists, skipping")
        return
    print("Building RAG knowledge base...")
    subprocess.run(["python", "api/llm/rag.py"], check=True)

if __name__ == "__main__":
    download_data()
    run_dbt()
    train_model()
    build_rag()
    print("Setup complete!")