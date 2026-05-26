import pandas as pd
import urllib.parse

params = urllib.parse.urlencode({
    "$limit": 500000,
    "$where": "created_date >= '2022-01-01'",
    "$order": "created_date DESC"
})

url = f"https://data.cityofnewyork.us/resource/erm2-nwe9.csv?{params}"

print("Downloading... (2-3 min)")
df = pd.read_csv(url)
df.to_parquet("data/raw/nyc311.parquet", index=False)
print(f"Done: {len(df):,} rows, {df.shape[1]} columns")
print(df.columns.tolist())
