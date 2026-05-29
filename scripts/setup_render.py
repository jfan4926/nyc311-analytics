import os
import re
import subprocess
import urllib.parse

import pandas as pd

DB_DIR  = "data/processed"
RAW_DIR = "data/raw"

DATE_COLS = {
    'created_date', 'closed_date',
    'due_date', 'resolution_action_updated_date'
}


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
    df  = pd.read_csv(url, low_memory=False)

    # Cast non-date object columns to str, but keep real NaN as None
    # so DuckDB can cast date columns to TIMESTAMP correctly
    for col in df.select_dtypes(include='object').columns:
        if col not in DATE_COLS:
            # Replace float NaN with None first, then cast to str
            df[col] = df[col].where(df[col].notna(), other=None)
            df[col] = df[col].apply(
                lambda x: str(x) if x is not None else None
            )

    df.to_parquet(f"{RAW_DIR}/nyc311.parquet", index=False)
    print(f"Downloaded {len(df):,} rows")


def run_dbt():
    os.makedirs(DB_DIR, exist_ok=True)
    workspace = os.path.abspath(".").replace("\\", "/")

    # Write profiles.yml with absolute paths
    profiles = f"""nyc311:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: "{workspace}/data/processed/nyc311.duckdb"
      threads: 2
      external_root: "{workspace}"
"""
    with open("dbt_project/nyc311/profiles.yml", "w") as f:
        f.write(profiles)

    # Update raw_data_path var in dbt_project.yml (idempotent regex)
    yml_path = "dbt_project/nyc311/dbt_project.yml"
    with open(yml_path) as f:
        content = f.read()

    new_content = re.sub(
        r'raw_data_path:.*',
        f'raw_data_path: "{workspace}/data/raw"',
        content
    )
    with open(yml_path, "w") as f:
        f.write(new_content)

    print("Running dbt...")
    result = subprocess.run(
        ["dbt", "run", "--profiles-dir", "."],
        cwd="dbt_project/nyc311",
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("=== dbt stderr ===")
        print(result.stderr)
        raise RuntimeError("dbt run failed")

    # Run dbt tests (non-fatal: log but don't abort)
    test = subprocess.run(
        ["dbt", "test", "--profiles-dir", "."],
        cwd="dbt_project/nyc311",
        capture_output=True,
        text=True
    )
    print(test.stdout)
    if test.returncode != 0:
        print("Warning: dbt tests failed (non-fatal)")
        print(test.stderr)

def train_model():
    if os.path.exists("ml_pipeline/models/model.pkl"):
        print("Model already exists, skipping training")
        return
    print("Training model...")
    os.makedirs("ml_pipeline/models", exist_ok=True)
    os.makedirs("ml_pipeline/data", exist_ok=True)
    result = subprocess.run(
        ["python", "ml_pipeline/train.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Model training failed")

# def train_model():
#     if os.path.exists("ml_pipeline/models/encoders.pkl"):
#         print("Model already exists, skipping training")
#         return
#     print("Training model (this may take a few minutes)...")
#     os.makedirs("ml_pipeline/models", exist_ok=True)
#     os.makedirs("ml_pipeline/data", exist_ok=True)
#     result = subprocess.run(
#         ["python", "ml_pipeline/train.py"],
#         capture_output=True,
#         text=True
#     )
#     print(result.stdout)
#     if result.returncode != 0:
#         print("=== train stderr ===")
#         print(result.stderr)
#         raise RuntimeError("Model training failed")


def build_rag():
    if os.path.exists("api/llm/chroma_store"):
        print("RAG store already exists, skipping")
        return
    print("Building RAG knowledge base...")
    result = subprocess.run(
        ["python", "api/llm/rag.py"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("=== rag stderr ===")
        print(result.stderr)
        raise RuntimeError("RAG build failed")


if __name__ == "__main__":
    download_data()
    run_dbt()
    train_model()
    build_rag()
    print("Setup complete!")