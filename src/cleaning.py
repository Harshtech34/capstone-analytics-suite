# src/cleaning.py
import pandas as pd
from src.logger import logger

def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    for c in ['Price','Quantity','Total_Sales']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'Total_Sales' not in df.columns or df['Total_Sales'].isna().any():
        if 'Price' in df.columns and 'Quantity' in df.columns:
            df['Total_Sales'] = (df['Price'].fillna(0) * df['Quantity'].fillna(0))
    df = df.dropna(subset=['Date','Product'])
    logger.info("Cleaned sales rows: %s", len(df))
    return df

def clean_house(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ['Area','Bedrooms','Bathrooms','Age','Price']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Area','Price'])
    logger.info("Cleaned house rows: %s", len(df))
    return df

def clean_churn(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ['Tenure','MonthlyCharges','TotalCharges','SeniorCitizen']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'TotalCharges' in df.columns and 'MonthlyCharges' in df.columns and 'Tenure' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['Tenure'])
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].astype(int)
    logger.info("Cleaned churn rows: %s", len(df))
    return df
