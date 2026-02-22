# src/data_loader.py
from pathlib import Path
import pandas as pd
from src.config import settings
from src.logger import logger

ROOT = Path(__file__).resolve().parents[1]

def raw_path(fname: str) -> Path:
    p = ROOT / settings.DATA_DIR / "raw" / fname
    if not p.exists():
        logger.error("File not found: %s", p)
        raise FileNotFoundError(p)
    return p

def load_sales(path: str = None) -> pd.DataFrame:
    p = raw_path(path or "sales_data.csv")
    df = pd.read_csv(p, parse_dates=["Date"], dayfirst=False)
    logger.info("Loaded sales: %s rows", len(df))
    return df

def load_house_prices(path: str = None) -> pd.DataFrame:
    p = raw_path(path or "house_prices.csv")
    df = pd.read_csv(p)
    logger.info("Loaded house prices: %s rows", len(df))
    return df

def load_churn(path: str = None) -> pd.DataFrame:
    p = raw_path(path or "customer_churn.csv")
    df = pd.read_csv(p)
    logger.info("Loaded churn: %s rows", len(df))
    return df

def save_processed(df, name: str):
    out = ROOT / settings.DATA_DIR / "processed"
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    df.to_csv(p, index=False)
    logger.info("Saved processed: %s", p)
