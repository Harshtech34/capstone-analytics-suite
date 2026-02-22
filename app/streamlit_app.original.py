# app/streamlit_app.py
"""
Production-ready Streamlit app entrypoint.

- Caching helpers (st.cache_resource, st.cache_data)
- Persistent geocode cache (data/processed/geocode_cache.csv)
- Startup-grade house valuation tab with local comps
- Sales and Churn tabs (use src.predict functions)
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------- Project root / PYTHON PATH --------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# -------------------- Imports from src --------------------
from src.predict import predict_house_price, predict_churn, predict_sales
from src.geocode import geocode_address  # low-level geocode call (calls Google)
from src.local_analysis import local_adjustment  # existing local adjustment util

# Optional map rendering
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False

# -------------------- App config --------------------
st.set_page_config(page_title="Capstone — E-Commerce, Real Estate & Churn", layout="wide")
logger = logging.getLogger("capstone")
logger.setLevel(logging.INFO)

# Basic CSS / theme tweaks
st.markdown(
    """
    <style>
    [data-testid="stToolbar"] {display: none;}
    .header {font-size: 26px; font-weight: 700;}
    .metric-label {font-size:14px;}
    .main {max-width:1200px; margin: auto;}
    .card {background: #ffffff; padding: 16px; border-radius: 8px; box-shadow: 0 3px 10px rgba(0,0,0,0.08);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚀 Capstone — E-Commerce, Real Estate & Churn")

# -------------------- CACHING HELPERS (top-level) --------------------
# Use st.cache_resource for heavy resources (models, large objects)
# Use st.cache_data for small I/O cached results (geocode cache read/write wrapper)

@st.cache_resource
def load_house_model_cached():
    """Load heavy house model once and reuse across reruns.
    Returns joblib pipeline or None.
    """
    model_path = Path(BASE_DIR) / "models" / "house_price_model.joblib"
    if model_path.exists():
        try:
            m = joblib.load(model_path)
            logger.info("Loaded house model from %s", model_path)
            return m
        except Exception as e:
            logger.exception("Failed to load house model: %s", e)
            return None
    else:
        logger.warning("House model not found at %s", model_path)
        return None

@st.cache_data(ttl=60 * 60 * 24)
def load_historical_sales_cached():
    """Load processed historical sales (cached 24h)."""
    hist_path = Path(BASE_DIR) / "data" / "processed" / "historical_sales.csv"
    if not hist_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(hist_path, parse_dates=["Date"], low_memory=False)
        return df
    except Exception as e:
        logger.exception("Failed to read historical_sales.csv: %s", e)
        return pd.DataFrame()

# Persistent geocode cache file (on disk)
GEOCODE_CACHE_PATH = Path(BASE_DIR) / "data" / "processed" / "geocode_cache.csv"
GEOCODE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

def _read_geocode_cache_df():
    """Read persistent cache from CSV (not cached by Streamlit; used inside cached_geocode)."""
    if GEOCODE_CACHE_PATH.exists():
        try:
            df = pd.read_csv(GEOCODE_CACHE_PATH, dtype={"address": str})
            return df
        except Exception as e:
            logger.exception("Failed to read geocode cache CSV: %s", e)
            return pd.DataFrame(columns=["address", "lat", "lon"])
    else:
        return pd.DataFrame(columns=["address", "lat", "lon"])

def _append_geocode_cache_df(address, lat, lon):
    """Append a single row to persistent cache safely."""
    try:
        cache_df = _read_geocode_cache_df()
        # Avoid duplicates
        if not cache_df[cache_df["address"] == address].empty:
            return
        new = pd.DataFrame([{"address": address, "lat": lat, "lon": lon}])
        updated = pd.concat([cache_df, new], ignore_index=True)
        updated.to_csv(GEOCODE_CACHE_PATH, index=False)
    except Exception as e:
        logger.exception("Failed to append geocode cache: %s", e)

@st.cache_data(ttl=60 * 60 * 24)
def cached_geocode(address: str):
    """
    Cached geocode wrapper:
    1) check persistent CSV cache
    2) if not found, call low-level geocode_address (Google)
    3) persist result to CSV and return lat, lon
    TTL 24h in-memory cache to avoid repeated calls during rapid reruns.
    """
    if not address or not str(address).strip():
        return None, None

    # check persistent cache (disk)
    cache_df = _read_geocode_cache_df()
    match = cache_df[cache_df["address"] == address]
    if not match.empty:
        try:
            lat = float(match.iloc[0]["lat"])
            lon = float(match.iloc[0]["lon"])
            logger.info("Geocode cache hit for address")
            return lat, lon
        except Exception:
            pass

    # Not cached -> call geocode (may call Google API)
    try:
        lat, lon = geocode_address(address)
    except Exception as e:
        logger.exception("geocode_address() failed: %s", e)
        lat, lon = None, None

    # persist if available
    if lat is not None and lon is not None:
        try:
            _append_geocode_cache_df(address, lat, lon)
        except Exception as e:
            logger.exception("Failed to persist geocode result: %s", e)

    return lat, lon

# -------------------- UTILS --------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Return haversine distance in kilometers."""
    import math
    if any(v is None or pd.isna(v) for v in (lat1, lon1, lat2, lon2)):
        return np.nan
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return 6371 * c

# -------------------- Layout & Tabs --------------------
tabs = st.tabs(["🏠 House Price", "🛒 E-Commerce Sales", "📉 Customer Churn"])

# ---------------------- HOUSE TAB (Upgraded with caching) ----------------------
with tabs[0]:
    st.markdown("## 🏠 AI Property Valuation Engine")
    st.markdown('<div class="card">Global ML valuation + hyperlocal adjustment using recent comps.</div>', unsafe_allow_html=True)

    input_col, result_col = st.columns([2, 1])

    with input_col:
        with st.form("house_form"):
            st.markdown("### Property Inputs")
            address = st.text_input("Full address (optional — improves local adjustment)", value="")
            col_a, col_b = st.columns(2)
            with col_a:
                area = st.number_input("Area (sqft)", min_value=1, max_value=1_000_000, value=1500, step=1)
                bedrooms = st.number_input("Bedrooms", min_value=0, max_value=50, value=3, step=1)
                age = st.number_input("Age (years)", min_value=0, max_value=200, value=5, step=1)
            with col_b:
                bathrooms = st.number_input("Bathrooms", min_value=0, max_value=50, value=2, step=1)
                property_type = st.selectbox("Property type", ["House", "Apartment", "Villa"])
                radius_km = st.slider("Comps radius (km)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
            months = st.slider("Market lookback (months)", min_value=1, max_value=12, value=3, step=1)
            submitted = st.form_submit_button("🚀 Run valuation")

    if submitted:
        with st.spinner("Running valuation engine..."):
            # 1) Geocode (cached)
            lat = lon = None
            if address and address.strip():
                lat, lon = cached_geocode(address)
                if lat is None or lon is None:
                    st.warning("Could not geocode address. App will run without local comps.")

            # 2) Base model prediction (the predict_house_price loads model internally)
            input_data = {
                "Area": area,
                "Bedrooms": bedrooms,
                "Bathrooms": bathrooms,
                "Age": age,
                "Property_Type": property_type,
            }
            base_price = None
            try:
                base_price = predict_house_price(input_data)
            except Exception as e:
                logger.exception("predict_house_price failed: %s", e)
                st.error("Base model prediction failed. Check logs.")
            
            # 3) Local adjustment using historical comps (cached load)
            hist = load_historical_sales_cached()
            adjusted_price = base_price
            factor = 1.0
            comps = pd.DataFrame()

            if lat is not None and lon is not None and not hist.empty and base_price is not None:
                # ensure necessary columns exist
                if {"lat", "lon", "Price", "Area", "Date"}.issubset(set(hist.columns)):
                    hist = hist.dropna(subset=["lat", "lon", "Price", "Area", "Date"])
                    cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days=30*months))
                    # compute haversine distance
                    hist["dist_km"] = hist.apply(lambda r: haversine_km(lat, lon, r["lat"], r["lon"]), axis=1)
                    recent = hist[(hist["Date"] >= cutoff) & (hist["dist_km"] <= radius_km)].copy()

                    if not recent.empty:
                        # Try to compute model-based predicted price for comps using cached pipeline
                        house_pipe = load_house_model_cached()
                        if house_pipe is not None:
                            # build feature DataFrame for pipeline; attempt mapping
                            def build_feat(row):
                                return {
                                    "Area": row.get("Area") if "Area" in row.index else row.get("area", np.nan),
                                    "Bedrooms": row.get("Bedrooms", np.nan),
                                    "Bathrooms": row.get("Bathrooms", np.nan),
                                    "Age": row.get("Age", np.nan),
                                    "Property_Type": row.get("Property_Type", ""),
                                    "Location": row.get("Location", "")
                                }
                            X = pd.DataFrame([build_feat(r) for _, r in recent.iterrows()])
                            try:
                                preds = house_pipe.predict(X)
                                recent = recent.reset_index(drop=True)
                                recent["pred_price"] = preds
                                recent["ratio_actual_pred"] = recent["Price"] / (recent["pred_price"] + 1e-9)
                                factor = float(recent["ratio_actual_pred"].median())
                                if np.isfinite(factor) and factor > 0:
                                    adjusted_price = base_price * factor
                                else:
                                    factor = 1.0
                                    adjusted_price = base_price
                                comps = recent
                            except Exception as e:
                                logger.exception("Failed predicting comps with pipeline: %s", e)
                                # fallback to median-price adjustment
                                med = recent["Price"].median()
                                if med and base_price:
                                    factor = med / base_price
                                    adjusted_price = base_price * factor
                                    comps = recent
                        else:
                            # fallback median-based adjustment
                            med = recent["Price"].median()
                            if med and base_price:
                                factor = med / base_price
                                adjusted_price = base_price * factor
                                comps = recent
                    else:
                        st.info("No recent comps found within radius/time window.")
                else:
                    st.warning("Historical data missing required columns (lat, lon, Price, Area, Date).")
            else:
                if hist.empty:
                    st.info("No historical sales data found (data/processed/historical_sales.csv).")
                elif base_price is None:
                    st.info("Base prediction unavailable; local adjustment skipped.")

            # 4) Results panel
            result_col1, result_col2 = st.columns([2, 1])
            with result_col1:
                st.subheader("Valuation Results")
                if base_price is not None:
                    st.metric("Base ML Valuation", f"₹{base_price:,.0f}")
                    st.metric("Market-adjusted Valuation", f"₹{adjusted_price:,.0f}")
                    st.caption(f"Adjustment factor (median actual/pred): {factor:.3f}")
                    confidence = "High" if len(comps) >= 10 else "Medium" if len(comps) >= 5 else "Low"
                    st.markdown(f"**Confidence:** {confidence} — based on {len(comps)} local comps")
                else:
                    st.error("Base model prediction failed.")

                if not comps.empty:
                    st.markdown("### Local Comparable Sales")
                    try:
                        import plotly.express as px
                        fig = px.box(comps, y="Price", title="Local comparable sale price distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.write(comps[["Date", "Price", "Area", "Bedrooms", "Bathrooms"]].head(10))

                    st.dataframe(comps.sort_values("Date", ascending=False).head(20))
                    csv = comps.to_csv(index=False)
                    st.download_button("Download comps CSV", csv, "comps.csv")
                else:
                    st.info("No comps to show.")

            with result_col2:
                st.subheader("Map")
                if FOLIUM_AVAILABLE and (lat is not None and lon is not None):
                    m = folium.Map(location=[lat, lon], zoom_start=14)
                    folium.Marker([lat, lon], tooltip="Target", icon=folium.Icon(color="red")).add_to(m)
                    if not comps.empty:
                        for _, r in comps.iterrows():
                            folium.CircleMarker(location=[r["lat"], r["lon"]],
                                                radius=5,
                                                tooltip=f"₹{int(r['Price']):,}",
                                                color="blue",
                                                fill=True).add_to(m)
                    st_folium(m, width=350, height=450)
                else:
                    if lat is not None and lon is not None:
                        st.map(pd.DataFrame([{"lat": lat, "lon": lon}]))
                    else:
                        st.info("No location to show. Provide address for map and local comps.")

# ---------------------- SALES TAB ----------------------
with tabs[1]:
    st.header("🛒 E-Commerce Sales Prediction")
    with st.form("sales_form"):
        product = st.selectbox("Product (informational)", ["Phone", "Laptop", "Headphones", "Tablet", "Monitor"])
        quantity = st.number_input("Quantity (units)", 1, 10000, 10)
        avg_price = st.number_input("Average price per unit (₹)", 1.0, 10_000_000.0, 1000.0, step=1.0)
        submitted2 = st.form_submit_button("Predict Sales")
        if submitted2:
            try:
                total_sales = predict_sales(int(quantity), float(avg_price))
                st.metric("🛒 Predicted Total Sales", f"₹{total_sales:,.2f}")
            except Exception as e:
                logger.exception("Sales prediction failed: %s", e)
                st.error("Sales prediction failed. Check logs.")

# ---------------------- CHURN TAB ----------------------
with tabs[2]:
    st.header("📉 Customer Churn Prediction")
    with st.form("churn_form"):
        tenure = st.number_input("Tenure (months)", 0, 1000, 12)
        monthly_charges = st.number_input("Monthly Charges (₹)", 0.0, 1_000_000.0, 100.0, step=1.0)
        total_charges = st.number_input("Total Charges (₹)", 0.0, 100_000_000.0, 1200.0, step=1.0)
        senior = st.selectbox("Senior Citizen", [0, 1])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method", ["Credit Card", "Electronic Check", "Bank Transfer"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        submitted3 = st.form_submit_button("Predict Churn")
        if submitted3:
            try:
                input_churn = {
                    "Tenure": tenure,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                    "SeniorCitizen": senior,
                    "Contract": contract,
                    "PaymentMethod": payment,
                    "PaperlessBilling": paperless
                }
                pred, proba = predict_churn(input_churn)
                st.metric("📉 Churn (Yes=1, No=0)", f"{pred}")
                if proba is not None:
                    st.metric("Churn probability", f"{proba:.1%}")
            except Exception as e:
                logger.exception("Churn prediction failed: %s", e)
                st.error("Churn prediction failed. Check logs.")
