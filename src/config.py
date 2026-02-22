# src/config.py
from pathlib import Path
import os

class Settings:
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODEL_DIR = "models"

    # Optional: environment overrides
    ENV = os.getenv("ENV", "development")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

settings = Settings()
