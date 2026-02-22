import pandas as pd
from pathlib import Path
from math import radians, cos, sin, asin, sqrt
import logging

logger = logging.getLogger("capstone")
ROOT = Path(__file__).resolve().parents[1]

def haversine(lat1, lon1, lat2, lon2):
    """Distance in km between two points."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

def local_adjustment(lat, lon, radius_km=2, months=3, historical_file="data/processed/historical_sales.csv"):
    """
    Adjust model prediction based on nearby recent sales
    """
    hist_path = ROOT / historical_file
    if not hist_path.exists():
        logger.warning(f"Historical sales file not found: {hist_path}")
        return 1.0  # default factor
    
    df = pd.read_csv(hist_path)
    df = df.dropna(subset=["lat", "lon", "Price", "Date"])
    df["distance_km"] = df.apply(lambda row: haversine(lat, lon, row["lat"], row["lon"]), axis=1)
    
    # Filter by distance and recent months
    df["Date"] = pd.to_datetime(df["Date"])
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=months)
    df_filtered = df[(df["distance_km"] <= radius_km) & (df["Date"] >= cutoff)]
    
    if df_filtered.empty:
        logger.info("No local comps found, using base model prediction")
        return 1.0
    
    # Compute adjustment factor
    factor = df_filtered["Price"].median() / df_filtered["Price"].mean()
    logger.info(f"Local adjustment factor: {factor:.3f} based on {len(df_filtered)} comps")
    return factor
