# src/test_models.py
import joblib
from pathlib import Path
import json
import pandas as pd
from src.config import settings
from src.logger import logger

MODELS_DIR = Path(__file__).resolve().parents[1] / settings.MODEL_DIR

def test_load(path):
    print("Testing:", path)
    m = joblib.load(path)
    print("Loaded type:", type(m))

def test_house():
    path = MODELS_DIR / "house_price_model.joblib"
    meta = json.loads((MODELS_DIR / "house_price_metadata.json").read_text())
    sample = {}
    for n in meta['numeric']:
        sample[n] = 1500 if n=='Area' else 3
    for c in meta['categorical']:
        sample[c] = "Suburb"
    df = pd.DataFrame([sample])
    m = joblib.load(path)
    pred = m.predict(df)[0]
    print("House sample pred:", pred)

def test_churn():
    path = MODELS_DIR / "churn_model.joblib"
    meta = json.loads((MODELS_DIR / "churn_metadata.json").read_text())
    sample = {}
    for n in meta['numeric']:
        sample[n] = 12 if n=='Tenure' else 100.0
    for c in meta['categorical']:
        sample[c] = "Month-to-month"
    df = pd.DataFrame([sample])
    m = joblib.load(path)
    pred = m.predict(df)[0]
    proba = m.predict_proba(df)[0,1] if hasattr(m, "predict_proba") else None
    print("Churn sample pred:", pred, "prob:", proba)

def test_sales():
    path = MODELS_DIR / "sales_model.joblib"
    m = joblib.load(path)
    df = pd.DataFrame([{"quantity": 10, "avg_price": 20000}])
    pred = m.predict(df)[0]
    print("Sales sample pred:", pred)

if __name__ == "__main__":
    try:
        test_load(MODELS_DIR / "house_price_model.joblib")
        test_house()
    except Exception as e:
        print("House test failed:", e)
    try:
        test_load(MODELS_DIR / "churn_model.joblib")
        test_churn()
    except Exception as e:
        print("Churn test failed:", e)
    try:
        test_load(MODELS_DIR / "sales_model.joblib")
        test_sales()
    except Exception as e:
        print("Sales test failed:", e)
    print("Tests finished.")
