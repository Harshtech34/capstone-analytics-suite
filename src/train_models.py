# src/train_models.py
"""
Robust training script for house price, churn and sales models.
Writes:
 - models/house_price_model.joblib
 - models/churn_model.joblib
 - models/sales_model.joblib
 - models/*_metadata.json
"""

import json
from pathlib import Path
import traceback
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report

# Local imports (expects src/ layout)
from src.data_loader import load_house_prices, load_churn, load_sales, save_processed
from src.cleaning import clean_house, clean_churn, clean_sales
from src.features import make_sales_monthly
from src.config import settings
from src.logger import logger

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / settings.MODEL_DIR
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def safe_run(fn, name):
    try:
        logger.info("Starting: %s", name)
        fn()
        logger.info("Finished: %s", name)
    except Exception as e:
        logger.error("ERROR in %s: %s", name, e)
        traceback.print_exc()

def train_house_price():
    # Load & clean
    df = load_house_prices()
    df = clean_house(df)
    if df.empty:
        raise RuntimeError("House prices data is empty after cleaning.")
    # select features
    numeric_feats = [c for c in ['Area','Bedrooms','Bathrooms','Age'] if c in df.columns]
    cat_feats = [c for c in ['Location','Property_Type'] if c in df.columns]
    X = df[numeric_feats + cat_feats].copy()
    y = df['Price']

    # pipelines
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])
    preproc = ColumnTransformer([('num', num_pipe, numeric_feats), ('cat', cat_pipe, cat_feats)], remainder='drop')

    pipeline = Pipeline([('preproc', preproc), ('model', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    logger.info("House model: R2=%.4f MAE=%.2f", r2_score(y_test, preds), mean_absolute_error(y_test, preds))

    # Save
    p = MODELS_DIR / "house_price_model.joblib"
    joblib.dump(pipeline, p)
    meta = {"numeric": numeric_feats, "categorical": cat_feats}
    (MODELS_DIR / "house_price_metadata.json").write_text(json.dumps(meta))
    logger.info("Saved house model -> %s", p)

def train_churn():
    df = load_churn()
    df = clean_churn(df)
    if df.empty:
        raise RuntimeError("Churn data is empty after cleaning.")
    numeric_feats = [c for c in ['Tenure','MonthlyCharges','TotalCharges','SeniorCitizen'] if c in df.columns]
    categorical_feats = [c for c in ['Contract','PaymentMethod','PaperlessBilling'] if c in df.columns]
    X = df[numeric_feats + categorical_feats].copy()
    y = df['Churn']

    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])
    preproc = ColumnTransformer([('num', num_pipe, numeric_feats), ('cat', cat_pipe, categorical_feats)], remainder='drop')

    pipeline = Pipeline([('preproc', preproc), ('model', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    logger.info("Churn model: accuracy=%.4f", accuracy_score(y_test, preds))
    logger.info("Churn classification report:\n%s", classification_report(y_test, preds))

    p = MODELS_DIR / "churn_model.joblib"
    joblib.dump(pipeline, p)
    meta = {"numeric": numeric_feats, "categorical": categorical_feats}
    (MODELS_DIR / "churn_metadata.json").write_text(json.dumps(meta))
    logger.info("Saved churn model -> %s", p)

def train_sales():
    df = load_sales()
    df = clean_sales(df)
    if df.empty:
        raise RuntimeError("Sales data is empty after cleaning.")
    agg = make_sales_monthly(df)
    # expected columns: quantity, avg_price, total_sales
    if not {'quantity','avg_price','total_sales'}.issubset(set(agg.columns)):
        raise RuntimeError("Sales aggregated columns missing: " + ", ".join(agg.columns))
    X = agg[['quantity','avg_price']].copy()
    y = agg['total_sales']

    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    preproc = ColumnTransformer([('num', num_pipe, ['quantity','avg_price'])], remainder='drop')

    pipeline = Pipeline([('preproc', preproc), ('model', RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    logger.info("Sales model: R2=%.4f MAE=%.2f", r2_score(y_test, preds), mean_absolute_error(y_test, preds))

    p = MODELS_DIR / "sales_model.joblib"
    joblib.dump(pipeline, p)
    meta = {"numeric": ['quantity','avg_price']}
    (MODELS_DIR / "sales_metadata.json").write_text(json.dumps(meta))
    logger.info("Saved sales model -> %s", p)

def main():
    safe_run(train_house_price, "train_house_price")
    safe_run(train_churn, "train_churn")
    safe_run(train_sales, "train_sales")
    logger.info("Training script finished. Check the models/ folder.")

if __name__ == "__main__":
    main()
