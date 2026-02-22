# src/predict.py
import joblib
import json
import pandas as pd
from pathlib import Path
from src.config import settings
from src.logger import logger

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / settings.MODEL_DIR

def _load_model(name):
    path = MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)

def _load_meta(name):
    meta_path = MODELS_DIR / name
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    return json.loads(meta_path.read_text())

def predict_house_price(input_dict):
    model = _load_model("house_price_model.joblib")
    meta = _load_meta("house_price_metadata.json")
    data = {}
    for n in meta['numeric']:
        data[n] = [input_dict.get(n)]
    for c in meta['categorical']:
        data[c] = [input_dict.get(c)]
    df = pd.DataFrame(data)
    pred = model.predict(df)[0]
    logger.info("House price predicted: %s", pred)
    return float(pred)

def predict_churn(input_dict):
    model = _load_model("churn_model.joblib")
    meta = _load_meta("churn_metadata.json")
    data = {}
    for n in meta['numeric']:
        data[n] = [input_dict.get(n)]
    for c in meta['categorical']:
        data[c] = [input_dict.get(c)]
    df = pd.DataFrame(data)
    pred = model.predict(df)[0]
    proba = float(model.predict_proba(df)[0,1]) if hasattr(model, "predict_proba") else None
    logger.info("Churn predicted: %s prob=%s", pred, proba)
    return int(pred), proba

def predict_sales(quantity: int, avg_price: float):
    model = _load_model("sales_model.joblib")
    df = pd.DataFrame([{"quantity": quantity, "avg_price": avg_price}])
    pred = model.predict(df)[0]
    logger.info("Sales predicted: %s", pred)
    return float(pred)
