# scripts/ingest_recent_sales.py
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "processed" / "historical_sales.csv"

def ingest(file_or_df):
    if isinstance(file_or_df, str):
        df_new = pd.read_csv(file_or_df)
    else:
        df_new = file_or_df
    # standardize columns: Price, Area, Bedrooms, Bathrooms, Date, Address, lat, lon
    # parse Date
    if 'Date' in df_new.columns:
        df_new['Date'] = pd.to_datetime(df_new['Date'])
    # append and dedupe on Address+Date or unique id
    if HIST.exists():
        df_hist = pd.read_csv(HIST, parse_dates=['Date'])
        combined = pd.concat([df_hist, df_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=['Address', 'Date'], keep='last')
    else:
        combined = df_new
    combined.to_csv(HIST, index=False)
    print("Ingested and saved", HIST)

if __name__ == "__main__":
    # sample usage: ingest a new CSV placed in data/raw/new_sales.csv
    import sys
    if len(sys.argv) > 1:
        ingest(sys.argv[1])
    else:
        print("Usage: python ingest_recent_sales.py path/to/new.csv")
