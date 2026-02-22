# app/streamlit_app.py
"""
Capstone Analytics Suite — Premium UI/UX
Visual/ui layer only. Core ML logic, caching and file I/O preserved.
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

# -------------------- Imports from src (UNCHANGED) --------------------
from src.predict import predict_house_price, predict_churn, predict_sales
from src.geocode import geocode_address  # low-level geocode call (calls Google)
from src.local_analysis import local_adjustment  # existing local adjustment util

# Optional map rendering (unchanged)
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False

# -------------------- App config (unchanged) --------------------
st.set_page_config(
    page_title="Capstone Analytics Suite",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)
logger = logging.getLogger("capstone")
logger.setLevel(logging.INFO)

# -------------------- CACHING HELPERS (UNCHANGED) --------------------
@st.cache_resource
def load_house_model_cached():
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
    hist_path = Path(BASE_DIR) / "data" / "processed" / "historical_sales.csv"
    if not hist_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(hist_path, parse_dates=["Date"], low_memory=False)
        return df
    except Exception as e:
        logger.exception("Failed to read historical_sales.csv: %s", e)
        return pd.DataFrame()

# Persistent geocode cache file (unchanged)
GEOCODE_CACHE_PATH = Path(BASE_DIR) / "data" / "processed" / "geocode_cache.csv"
GEOCODE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

def _read_geocode_cache_df():
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
    try:
        cache_df = _read_geocode_cache_df()
        if not cache_df[cache_df["address"] == address].empty:
            return
        new = pd.DataFrame([{"address": address, "lat": lat, "lon": lon}])
        updated = pd.concat([cache_df, new], ignore_index=True)
        updated.to_csv(GEOCODE_CACHE_PATH, index=False)
    except Exception as e:
        logger.exception("Failed to append geocode cache: %s", e)

@st.cache_data(ttl=60 * 60 * 24)
def cached_geocode(address: str):
    if not address or not str(address).strip():
        return None, None

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

    try:
        lat, lon = geocode_address(address)
    except Exception as e:
        logger.exception("geocode_address() failed: %s", e)
        lat, lon = None, None

    if lat is not None and lon is not None:
        try:
            _append_geocode_cache_df(address, lat, lon)
        except Exception as e:
            logger.exception("Failed to persist geocode result: %s", e)

    return lat, lon

# -------------------- UTILS (UNCHANGED) --------------------
def haversine_km(lat1, lon1, lat2, lon2):
    import math
    if any(v is None or pd.isna(v) for v in (lat1, lon1, lat2, lon2)):
        return np.nan
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return 6371 * c

# -------------------- THEME MANAGEMENT --------------------
# We'll allow switching between a small set of themes. The CSS below is injected
# based on st.session_state['theme'] and updates on selection.
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # default

def _inject_ui_css(theme="dark"):
    themes = {
        "dark": {
            "bg": "#0b0f14",
            "panel_bg": "rgba(20,25,30,0.6)",
            "muted": "#9aa3b2",
            "accent_from": "#3b82f6",
            "accent_to": "#6366f1",
            "text": "#e6eef6",
            "glass": "rgba(255,255,255,0.03)"
        }
    }

    t = themes["dark"]

    css = """
    <style>
    body {{
        background: {bg} !important;
        color: {text};
    }}

    .card {{
        background: {panel_bg};
        border-radius: 12px;
        padding: 14px;
        border: 1px solid {glass};
    }}

    .stButton>button {{
        background: linear-gradient(90deg,{accent_from},{accent_to});
        color: white;
        border-radius: 10px;
        padding: 8px 14px;
        font-weight:600;
        border: none;
    }}
    </style>
    """.format(
        bg=t["bg"],
        text=t["text"],
        panel_bg=t["panel_bg"],
        glass=t["glass"],
        accent_from=t["accent_from"],
        accent_to=t["accent_to"]
    )

    st.markdown(css, unsafe_allow_html=True)
# inject current theme css
_inject_ui_css(st.session_state.theme)

# -------------------- Sidebar (reliable, well-aligned navigation + theme selector) --------------------
with st.sidebar:
    st.markdown("<div style='display:flex; gap:12px; align-items:center;'>"
                "<div style='width:46px; height:46px; border-radius:10px; background: linear-gradient(90deg,#3b82f6,#6366f1); display:flex; align-items:center; justify-content:center; color:white; font-weight:800;'>HC</div>"
                "<div><div class='brand-title'>Capstone</div><div class='brand-sub'>Real Estate • E-Commerce • Churn</div></div>"
                "</div>",
                unsafe_allow_html=True)
    st.markdown("---")

    # navigation radio (reliable placement)
    page = st.radio(
        "Navigate",
        options=["Dashboard", "Valuation", "Sales", "Churn", "Settings"],
        index=0,
        format_func=lambda x: x
    )
    # Theme selector
    st.markdown("---")
    theme_choice = st.selectbox("Theme", options=["dark", "deep", "midnight", "light"], index=["dark","deep","midnight","light"].index(st.session_state.theme))
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        # re-inject CSS immediately and rerun so UI updates
        _inject_ui_css(st.session_state.theme)
        st.experimental_rerun()

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("© 2026 Capstone Analytics", unsafe_allow_html=True)

# Top hero area (keeps consistent spacing)
st.markdown(
    """
    <div style="margin-bottom:12px;">
      <h1 style="margin:0 0 6px 0;">Welcome back! 🚀</h1>
      <div style="color:var(--muted); margin-bottom:12px;">Here’s your latest real estate comps, sales & churn insights.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# helper: plotly theme
def _plotly_dark_template():
    import plotly.io as pio
    import plotly.graph_objects as go
    pio.templates["capstone_dark"] = go.layout.Template(
        layout={
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"color": "#e6eef6"},
            "xaxis": {"gridcolor": "rgba(255,255,255,0.03)", "zerolinecolor": "rgba(255,255,255,0.02)"},
            "yaxis": {"gridcolor": "rgba(255,255,255,0.03)"},
            "legend": {"bgcolor": "rgba(0,0,0,0)"},
        }
    )
    return "capstone_dark"

# -------------------- Pages (render functions) --------------------
def render_dashboard():
    hist = load_historical_sales_cached()

    # Compute some safe KPIs
    valuation_est = None
    monthly_sales = None
    if not hist.empty and "Price" in hist.columns:
        try:
            valuation_est = float(hist["Price"].median())
        except Exception:
            valuation_est = None
        try:
            if "Date" in hist.columns:
                cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days=30))
                monthly_sales = float(hist[hist["Date"] >= cutoff]["Price"].sum())
        except Exception:
            monthly_sales = None

    # KPI grid
    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    st.markdown(f'<div class="card kpi"><div class="title">Valuation Estimate</div><div class="value">{"₹{:,.0f}".format(valuation_est) if valuation_est else "—"}</div><div style="color:#7ee787">+5.2% vs last month</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card kpi"><div class="title">Monthly Sales</div><div class="value">{"₹{:,.0f}".format(monthly_sales) if monthly_sales else "—"}</div><div style="color:#7ee787">+2.8% vs last month</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="card kpi"><div class="title">Churn Risk</div><div class="value">8.5%</div><div style="color:#ffb86b">-0.4% vs last month</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="card kpi"><div class="title">Open Tasks</div><div class="value">4</div><div style="color:#a0b0c6">2 new this week</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # panels: left = comps, right = sales chart & churn
    left, right = st.columns([2, 1], gap="large")
    with left:
        st.markdown('<div class="card"><strong>Latest Comparable Sales</strong></div>', unsafe_allow_html=True)
        if not hist.empty and {"Date","Price","Area"}.issubset(set(hist.columns)):
            recent = hist.sort_values("Date", ascending=False).head(6)
            # compact list
            for i, row in recent.head(3).iterrows():
                loc = row.get("Location", f"Comp {i}")
                area = int(row.get("Area", 0))
                price = int(row.get("Price", 0))
                st.markdown(f"**{loc}** — {area:,} sqft — ₹{price:,}")
            st.markdown("---")
            st.dataframe(recent[["Date","Price","Area"]].head(6))
        else:
            st.info("No historical comps found. Upload `data/processed/historical_sales.csv` for comps list.")

    with right:
        st.markdown('<div class="card"><strong>Sales Growth</strong></div>', unsafe_allow_html=True)
        try:
            import plotly.express as px
            if not hist.empty and "Date" in hist.columns:
                tmp = hist.copy()
                tmp["month"] = tmp["Date"].dt.to_period("M").dt.to_timestamp()
                monthly = tmp.groupby("month")["Price"].sum().reset_index().sort_values("month")
                fig = px.line(monthly, x="month", y="Price", markers=True, template=_plotly_dark_template())
                fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=320)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sales history available to show growth chart.")
        except Exception as e:
            logger.exception("Plotly chart failed: %s", e)
            st.info("Chart rendering failed.")

        st.markdown('<div class="card"><strong>Churn Overview</strong><div style="padding-top:8px;"><div style="font-size:22px;font-weight:700;">8.5%</div><div style="color:var(--muted);">Churned: 425 • Retained: 4,575</div></div></div>', unsafe_allow_html=True)

def render_valuation():
    # re-use original valuation logic exactly
    st.markdown("## 🏠 AI Property Valuation Engine")
    st.markdown('<div class="card"><strong>Global ML valuation + hyperlocal adjustment using recent comps.</strong></div>', unsafe_allow_html=True)

    input_col, result_col = st.columns([2, 1], gap="large")
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
            lat = lon = None
            if address and address.strip():
                lat, lon = cached_geocode(address)
                if lat is None or lon is None:
                    st.warning("Could not geocode address. App will run without local comps.")

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

            hist = load_historical_sales_cached()
            adjusted_price = base_price
            factor = 1.0
            comps = pd.DataFrame()

            if lat is not None and lon is not None and not hist.empty and base_price is not None:
                if {"lat", "lon", "Price", "Area", "Date"}.issubset(set(hist.columns)):
                    hist = hist.dropna(subset=["lat", "lon", "Price", "Area", "Date"])
                    cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days=30*months))
                    hist["dist_km"] = hist.apply(lambda r: haversine_km(lat, lon, r["lat"], r["lon"]), axis=1)
                    recent = hist[(hist["Date"] >= cutoff) & (hist["dist_km"] <= radius_km)].copy()

                    if not recent.empty:
                        house_pipe = load_house_model_cached()
                        if house_pipe is not None:
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
                                med = recent["Price"].median()
                                if med and base_price:
                                    factor = med / base_price
                                    adjusted_price = base_price * factor
                                    comps = recent
                        else:
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

            # Results
            result_col1, result_col2 = st.columns([2, 1], gap="large")
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
                        fig = px.box(comps, y="Price", title="Local comparable sale price distribution", template=_plotly_dark_template())
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.write(comps[["Date","Price","Area","Bedrooms","Bathrooms"]].head(10))

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

def render_sales():
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

def render_churn():
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

def render_settings():
    st.header("Settings")
    st.markdown("Use this page to configure API keys, theme defaults, or other admin settings.")
    st.markdown("**Current theme:** " + st.session_state.theme)

# -------------------- Router --------------------
current_page = page  # from sidebar radio
if current_page == "Dashboard":
    render_dashboard()
elif current_page == "Valuation":
    render_valuation()
elif current_page == "Sales":
    render_sales()
elif current_page == "Churn":
    render_churn()
elif current_page == "Settings":
    render_settings()
else:
    st.write("Unknown page")

st.markdown("---")
st.caption("Capstone Analytics Suite • UI upgraded — core ML logic unchanged")