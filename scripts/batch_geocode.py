# scripts/batch_geocode.py
import pandas as pd
import time
from pathlib import Path
from src.google_places import place_details
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

INPUT = ROOT / "data" / "raw" / "historical_sales_raw.csv"   # your raw historical dataset
OUTPUT = ROOT / "data" / "processed" / "historical_sales.csv"
CACHE = ROOT / "data" / "processed" / "geocode_cache.csv"

def main():
    df = pd.read_csv(INPUT)
    cache = pd.DataFrame()
    if CACHE.exists():
        cache = pd.read_csv(CACHE)
    cache_index = {row.address: (row.lat, row.lon) for _, row in cache.iterrows()} if not cache.empty else {}

    results = []
    for idx, r in df.iterrows():
        addr = r.get("Address") or r.get("address") or ""
        if not addr:
            continue
        if addr in cache_index:
            lat, lon = cache_index[addr]
        else:
            # use Places via text search fallback: use autocomplete -> take first suggestion -> place_details
            # But here we try directly place_details by providing place_id if available. For raw addresses, use geocode:
            detail = place_details(addr)
            if detail:
                lat, lon = detail["lat"], detail["lon"]
            else:
                lat, lon = None, None
            # cache it
            cache_index[addr] = (lat, lon)
            # be polite with API
            time.sleep(0.12)
        row = r.to_dict()
        row["lat"] = lat
        row["lon"] = lon
        results.append(row)

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT, index=False)
    # save cache
    cache_df = pd.DataFrame([{"address": k, "lat": v[0], "lon": v[1]} for k, v in cache_index.items()])
    cache_df.to_csv(CACHE, index=False)
    print("Saved processed historical to", OUTPUT)

if __name__ == "__main__":
    main()
