# app/streamlit_app.py
"""
Capstone Analytics Suite — Production-ready layout rewrite.

This file:
 - Preserves all original ML & I/O logic (predict_house_price, predict_churn, predict_sales, cached geocode, historical sales, etc.)
 - Rebuilds the UI layout into a true dashboard (wide layout, hero, KPI grid, main two-column content, clean footer)
 - Uses tokenized CSS injection to avoid Python string formatting issues and linter errors
 - Keeps folium/plotly usage unchanged where present (optional)
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import random
import json

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib

# -------------------- Project root / PYTHON PATH --------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# -------------------- Imports from src (UNCHANGED) --------------------
from src.predict import predict_house_price, predict_churn, predict_sales
from src.geocode import geocode_address
from src.local_analysis import local_adjustment

# Optional map rendering
try:
    import folium
    from streamlit_folium import st_folium
    from folium.plugins import MarkerCluster
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False

# -------------------- App config --------------------
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

# -------------------- THEME & TOKENS --------------------
THEMES = {
    "image_theme": {
        "bg": "#0b0f14",
        "panel": "rgba(18,22,26,0.52)",
        "muted": "#9aa3b2",
        "accent_from": "#2f6bff",
        "accent_to": "#23c7b8",
        "title": "#e6eef6",
        "glass": "rgba(255,255,255,0.03)"
    },
    "dark": {
        "bg": "#0b0f14",
        "panel": "rgba(20,25,30,0.85)",
        "muted": "#9aa3b2",
        "accent_from": "#06b6d4",
        "accent_to": "#2dd4bf",
        "title": "#e6eef6",
        "glass": "rgba(255,255,255,0.02)"
    },
    "midnight": {
        "bg": "#071018",
        "panel": "rgba(12,16,20,0.9)",
        "muted": "#9fb0c6",
        "accent_from": "#06b6d4",
        "accent_to": "#06d6a0",
        "title": "#eaf6f8",
        "glass": "rgba(255,255,255,0.02)"
    },
    "light": {
        "bg": "#f6f8fb",
        "panel": "white",
        "muted": "#6b7280",
        "accent_from": "#2563eb",
        "accent_to": "#06b6d4",
        "title": "#0b1220",
        "glass": "rgba(0,0,0,0.04)"
    }
}

if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "image_theme"

# -------------------- CLEAN CSS TEMPLATE (tokenized) --------------------
CSS_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Poppins:wght@600;700&display=swap');

/* Design tokens */
:root{
  --bg: __BG__;
  --panel: __PANEL__;
  --card-bg: rgba(255,255,255,0.02);
  --text-primary: __TITLE__;
  --text-muted: __MUTED__;
  --accent-from: __ACCENT_FROM__;
  --accent-to: __ACCENT_TO__;
  --accent-teal: #2dd4bf;
  --accent-cyan: #06b6d4;
  --accent-warning: #f59e0b;
  --accent-purple: #6d28d9;
  --glass: __GLASS__;
  --radius: 12px;
  --sidebar-w: 260px;
  --side-pad: 28px;
  --card-gap: 24px;
}

/* Page */
html, body, [class*="css"] {
  background: var(--bg) !important;
  color: var(--text-primary);
  font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}

/* Sidebar sizing & style */
[data-testid="stSidebar"] > div:first-child {
  width: var(--sidebar-w) !important;
  min-width: var(--sidebar-w) !important;
  max-width: var(--sidebar-w) !important;
  background: linear-gradient(180deg, rgba(6,8,10,0.98), rgba(12,14,20,0.96));
  padding: 24px;
  box-sizing: border-box;
  border-right: 1px solid rgba(255,255,255,0.02);
}

/* Main container */
.block-container {
  max-width: 1365px;
  padding-left: var(--side-pad);
  padding-right: var(--side-pad);
  padding-top: 26px;
  padding-bottom: 40px;
}

/* Brand */
.brand-title { font-family:'Poppins', sans-serif; font-size:18px; font-weight:700; color: var(--text-primary); }
.brand-sub { color: var(--text-muted); font-size: 12px; margin-top:4px; }

/* Hero */
h1 { font-family: 'Poppins', sans-serif; font-size:34px; letter-spacing:-0.4px; margin-bottom: 6px; color: var(--text-primary); }
.hero-lead { color: var(--text-muted); font-size:16px; margin-bottom: 18px; }

/* KPI grid */
.kpi-grid {
  display:grid;
  grid-template-columns: repeat(4, 1fr);
  gap: var(--card-gap);
  margin-bottom: 18px;
}
.kpi {
  position: relative;
  overflow: hidden;
  border-radius: var(--radius);
  padding: 18px;
  min-height: 120px;
  color: var(--text-primary);
  border: 1px solid var(--glass);
  box-shadow: 0 8px 30px rgba(2,6,23,0.6);
  background: linear-gradient(90deg, rgba(255,255,255,0.01), rgba(255,255,255,0.00));
  backdrop-filter: blur(6px);
  transition: transform .18s ease, box-shadow .18s ease;
}
.kpi:hover { transform: translateY(-6px); box-shadow: 0 20px 40px rgba(2,6,23,0.6); }
.kpi .tile { width:52px; height:52px; border-radius:12px; display:inline-flex; align-items:center; justify-content:center; margin-right:12px; box-shadow: inset 0 -6px 18px rgba(0,0,0,0.28); }
.kpi .title { color: var(--text-muted); font-size:13px; margin-bottom:6px; }
.kpi .value { font-size:28px; font-weight:800; margin-bottom:6px; }
.kpi .delta { font-size:13px; }

/* Card / panel style */
.card {
  border-radius: var(--radius);
  padding: 16px;
  border: 1px solid var(--glass);
  background: var(--panel);
  box-shadow: 0 12px 36px rgba(2,6,23,0.45);
  margin-bottom: 18px;
}

/* Comps list */
.comps-row {
  display:flex; align-items:center; justify-content:space-between;
  padding:10px; border-radius:10px; margin-bottom:8px;
  background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.00));
  transition: background .12s ease, transform .08s ease;
}
.comps-row:hover { background: rgba(255,255,255,0.02); transform: translateY(-2px); }

/* Plotly/Dataframe tweaks */
.plotly-graph-div, .stPlotlyChart > div { border-radius: 12px; overflow: hidden; }
.stDataFrame table td, .stDataFrame table th { color: var(--text-primary) !important; }

/* Responsive */
@media (max-width: 1100px) {
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 640px) {
  .kpi-grid { grid-template-columns: 1fr; }
  h1 { font-size: 28px; }
  [data-testid="stSidebar"] > div:first-child { display:none !important; }
}
</style>
"""

def inject_theme_css(theme_key):
    theme = THEMES.get(theme_key, THEMES["dark"])
    css = CSS_TEMPLATE.replace("__BG__", theme["bg"]) \
                      .replace("__PANEL__", theme["panel"]) \
                      .replace("__MUTED__", theme["muted"]) \
                      .replace("__ACCENT_FROM__", theme["accent_from"]) \
                      .replace("__ACCENT_TO__", theme["accent_to"]) \
                      .replace("__TITLE__", theme["title"]) \
                      .replace("__GLASS__", theme["glass"])
    st.markdown(css, unsafe_allow_html=True)

# initial injection
inject_theme_css(st.session_state.ui_theme)

# -------------------- Plotly dark template (unchanged) --------------------
def _plotly_dark_template():
    import plotly.io as pio
    import plotly.graph_objects as go
    pio.templates["capstone_dark"] = go.layout.Template(
        layout={
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"color": THEMES[st.session_state.ui_theme]["title"]},
            "xaxis": {"gridcolor": "rgba(255,255,255,0.03)", "zerolinecolor": "rgba(255,255,255,0.02)"},
            "yaxis": {"gridcolor": "rgba(255,255,255,0.03)"},
            "legend": {"bgcolor": "rgba(0,0,0,0)"},
        }
    )
    return "capstone_dark"

# ---------- BEGIN: Custom HTML / Hybrid visual helpers ----------
def render_custom_html_full(hist):
    """
    Option 1 — Full custom HTML dashboard rendered inside a single components.html.
    Visual-only: nice pixel control, Chart.js based charts, static map placeholder.
    """
    # Prepare safe data
    if hist is None or hist.empty:
        labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        values = [100, 120, 150, 170, 190, 210]
        comps = []
        val_median = 20000000
        monthly_total = 1200000
    else:
        try:
            tmp = hist.copy()
            tmp["month"] = tmp["Date"].dt.to_period("M").dt.to_timestamp()
            monthly = tmp.groupby("month")["Price"].sum().reset_index().sort_values("month").tail(6)
            labels = [d.strftime("%b %Y") for d in monthly["month"].tolist()]
            values = [int(v) for v in monthly["Price"].tolist()]
        except Exception:
            labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
            values = [100, 120, 150, 170, 190, 210]
        comps = hist.sort_values("Date", ascending=False).head(6).to_dict(orient="records")
        try:
            val_median = int(hist["Price"].median())
        except Exception:
            val_median = 0
        try:
            cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days=30))
            monthly_total = int(hist[hist["Date"] >= cutoff]["Price"].sum())
        except Exception:
            monthly_total = sum(values) if values else 0

    churn_pct = 8.5
    churned = 425
    retained = 4575
    num_comps = len(comps)

    js_labels = json.dumps(labels)
    js_values = json.dumps(values)
    js_comps = json.dumps(comps)

    # HTML template uses placeholders (__PLACEHOLDER__) replaced safely below
    HTML_TEMPLATE = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Capstone Analytics Preview</title>
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
      <style>
        :root{ --bg:#071018; --card:#0f1724; --muted:#9aa3b2; --accent:#06b6d4; --accent2:#2f6bff; --glass: rgba(255,255,255,0.03);}
        body{margin:0; font-family:Inter, sans-serif; background:var(--bg); color:#e6eef6; -webkit-font-smoothing:antialiased;}
        .shell{display:flex; height:100vh; width:100vw; align-items:stretch;}
        /* left sidebar visual (preview only) */
        .leftbar{width:260px; background:linear-gradient(180deg, #06070a, #0b0f14); padding:24px; box-sizing:border-box; border-right:1px solid rgba(255,255,255,0.02);}
        .logo{font-weight:800; font-family:Poppins, sans-serif; font-size:18px; color:#fff;}
        .sub{color:var(--muted); font-size:12px; margin-top:6px}
        .main{flex:1; overflow:auto; padding:28px; box-sizing:border-box;}
        .hero{display:flex; align-items:center; justify-content:space-between;}
        h1{font-family:Poppins, sans-serif; margin:0; font-size:34px;}
        .hero .lead{color:var(--muted); margin-top:4px; font-size:16px;}
        .kpi-row{display:grid; grid-template-columns: repeat(4, 1fr); gap:24px; margin-top:18px;}
        .kpi{background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:12px; padding:16px; min-height:120px; box-shadow:0 8px 30px rgba(2,6,23,0.6); border:1px solid rgba(255,255,255,0.02); backdrop-filter: blur(6px);}
        .kpi .label{color:var(--muted); font-size:12px;}
        .kpi .value{font-size:26px; font-weight:800; margin-top:6px;}
        .kpi .delta{font-size:13px; margin-top:6px;}
        .grid-main{display:grid; grid-template-columns: 2fr 1fr; gap:24px; margin-top:24px;}
        .card{background:var(--card); padding:14px; border-radius:12px; border:1px solid rgba(255,255,255,0.02); box-shadow:0 12px 36px rgba(2,6,23,0.45);}
        .map-stub{height:220px; border-radius:10px; background:linear-gradient(90deg, var(--accent), var(--accent2)); display:flex; align-items:center; justify-content:center; color:#001; font-weight:700;}
        .comps-list{margin-top:12px;}
        .comps-row{display:flex; align-items:center; justify-content:space-between; padding:10px; border-radius:10px; margin-bottom:8px; background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.00));}
        .comps-left{display:flex; gap:10px; align-items:center;}
        .muted{color:var(--muted); font-size:12px;}
        .table-footer{display:flex; justify-content:space-between; align-items:center; margin-top:12px;}
        .chart-legend{display:flex; gap:10px; align-items:center; margin-top:10px;}
        .legend-item{display:flex; gap:6px; align-items:center; font-size:12px; color:var(--muted);}
        .dot{width:10px; height:10px; border-radius:3px;}
        @media (max-width:900px){
          .kpi-row{grid-template-columns:repeat(2,1fr);}
          .grid-main{grid-template-columns:1fr;}
        }
      </style>
    </head>
    <body>
      <div class="shell">
        <div class="leftbar">
          <div class="logo">Capstone Analytics Suite</div>
          <div class="sub">Real Estate • E-Commerce • Churn</div>
        </div>

        <div class="main">
          <div class="hero">
            <div>
              <h1>Welcome back! 🚀</h1>
              <div class="lead">Here’s your latest real estate comps, sales & churn insights.</div>
            </div>
            <div style="text-align:right;">
              <button style="background:transparent;border:1px solid rgba(255,255,255,0.04); color:var(--muted); padding:8px 10px; border-radius:8px;">Deploy</button>
            </div>
          </div>

          <div class="kpi-row">
            <div class="kpi">
              <div class="label">Valuation Estimate</div>
              <div class="value">₹__VAL_MEDIAN__</div>
              <div class="delta" style="color:#7ee787;">+5.2% vs last month</div>
            </div>
            <div class="kpi">
              <div class="label">Monthly Sales</div>
              <div class="value">₹__MONTHLY_TOTAL__</div>
              <div class="delta" style="color:var(--accent-cyan);">+2.8% vs last month</div>
            </div>
            <div class="kpi" style="background: linear-gradient(90deg, rgba(245,158,11,0.06), rgba(255,255,255,0.00));">
              <div class="label">Churn Risk</div>
              <div class="value">__CHURN_PCT__%</div>
              <div class="delta" style="color:var(--accent-warning);">-0.4% vs last month</div>
            </div>
            <div class="kpi" style="background: linear-gradient(90deg, rgba(109,40,217,0.06), rgba(255,255,255,0.00));">
              <div class="label">Open Tasks</div>
              <div class="value">4</div>
              <div class="delta" style="color:#a0b0c6;">2 new this week</div>
            </div>
          </div>

          <div class="grid-main">
            <div>
              <div class="card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                  <div style="font-weight:600;">📍 Latest Comparable Sales</div>
                  <div class="muted">__NUM_COMPS__ items</div>
                </div>
                <div class="map-stub">MAP PLACEHOLDER</div>
                <div class="comps-list" id="comps-list">
                  <!-- rows injected by JS -->
                </div>

                <div class="card" style="margin-top:12px; padding:10px;">
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-weight:600;">Recent Sales</div>
                    <a href="#" style="color:var(--muted); font-size:13px;">View all comps</a>
                  </div>
                  <div style="margin-top:10px;">
                    <table style="width:100%; border-collapse:collapse; color:var(--text-primary);">
                      <thead style="color:var(--muted); font-size:12px;">
                        <tr><th style="text-align:left; padding:6px 8px;">Date</th><th style="text-align:left; padding:6px 8px;">Price</th><th style="text-align:left; padding:6px 8px;">Area</th><th style="text-align:left; padding:6px 8px;">Beds</th></tr>
                      </thead>
                      <tbody id="recent-table" style="font-size:13px;">
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>

            <div>
              <div class="card" style="margin-bottom:18px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                  <div style="font-weight:600;">📈 Sales Growth Summary <span style="color:var(--muted); font-size:13px;">+2.8%</span></div>
                  <select style="background:transparent; color:var(--muted); border:1px solid rgba(255,255,255,0.03); padding:6px 8px; border-radius:8px;">
                    <option>Last 6 months</option>
                    <option>Last 12 months</option>
                  </select>
                </div>
                <div style="margin-top:12px;">
                  <canvas id="salesChart" height="220"></canvas>
                </div>
                <div class="chart-legend">
                  <div class="legend-item"><span class="dot" style="background:#06b6d4;"></span> Phones</div>
                  <div class="legend-item"><span class="dot" style="background:#6d28d9;"></span> Laptops</div>
                  <div class="legend-item"><span class="dot" style="background:#f59e0b;"></span> Tablets</div>
                </div>
              </div>

              <div class="card">
                <div style="font-weight:600; margin-bottom:10px;">📉 Churn Overview</div>
                <div style="display:flex; align-items:center; gap:12px;">
                  <canvas id="donut" width="140" height="140"></canvas>
                  <div>
                    <div style="font-size:22px; font-weight:700;">__CHURN_PCT__% <span style="color:var(--accent-warning); font-size:14px; margin-left:8px;">-0.4%</span></div>
                    <div style="color:var(--muted); margin-top:8px;"><span style="display:inline-block;width:10px;height:10px;background:#fb923c;margin-right:8px;border-radius:2px;"></span> Churned: __CHURNED__</div>
                    <div style="color:var(--muted); margin-top:6px;"><span style="display:inline-block;width:10px;height:10px;background:#06b6d4;margin-right:8px;border-radius:2px;"></span> Retained: __RETAINED__</div>
                  </div>
                </div>
              </div>

            </div>
          </div>

        </div>
      </div>

      <script>
        const labels = __JS_LABELS__;
        const values = __JS_VALUES__;
        const comps = __JS_COMPS__ || [];
        const compsList = document.getElementById('comps-list');
        const recentTable = document.getElementById('recent-table');

        if (compsList) {
          if (comps.length) {
            comps.slice(0,6).forEach(function(c) {
              const row = document.createElement('div');
              row.className = 'comps-row';
              row.innerHTML = `<div class="comps-left"><div style="width:10px;height:10px;background:#06b6d4;border-radius:3px;"></div>
                <div>
                  <div style="font-weight:600;">${c.Location || c.Address || 'Comp'}</div>
                  <div class="muted">+${(Math.random()*6).toFixed(2)} • 4 days ago</div>
                </div></div>
                <div style="text-align:right;"><div style="font-weight:700;">${(c.Price ? '₹' + Number(c.Price).toLocaleString() : '-')}</div><div class="muted">${c.Area ? c.Area + ' sqft' : ''}</div></div>`;
              compsList.appendChild(row);
            });

            // small table rows
            comps.slice(0,6).forEach(function(c) {
              const tr = document.createElement('tr');
              tr.innerHTML = `<td style="padding:6px 8px;">${c.Date ? (new Date(c.Date)).toLocaleDateString() : '—'}</td>
                              <td style="padding:6px 8px;">${c.Price ? '₹' + Number(c.Price).toLocaleString() : '-'}</td>
                              <td style="padding:6px 8px;">${c.Area ? c.Area : '-'}</td>
                              <td style="padding:6px 8px;">${c.Bedrooms ? c.Bedrooms : '-'}</td>`;
              recentTable.appendChild(tr);
            });

          } else {
            compsList.innerHTML = "<div style='color:var(--muted);'>No comps available</div>";
          }
        }

        // Chart.js line with glow
        const ctx = document.getElementById('salesChart') && document.getElementById('salesChart').getContext ? document.getElementById('salesChart').getContext('2d') : null;
        if (ctx) {
          new Chart(ctx, {
            type: 'line',
            data: {
              labels: labels,
              datasets: [{
                label: 'Sales',
                data: values,
                borderColor: '#06b6d4',
                pointBackgroundColor: '#fff',
                pointBorderColor: '#06b6d4',
                pointRadius: 5,
                pointHoverRadius: 6,
                tension: 0.35,
                fill: false,
                borderWidth: 3,
                // shadow plugin style simulated via thicker semi-transparent line
              }]
            },
            options: {
              plugins: { legend: { display: false } },
              scales: {
                x: { grid: { display: false }, ticks: { color:'#9aa3b2' } },
                y: { grid: { color:'rgba(255,255,255,0.03)' }, ticks: { color:'#9aa3b2' } }
              }
            }
          });
        }

        // Donut chart
        const dctx = document.getElementById('donut') && document.getElementById('donut').getContext ? document.getElementById('donut').getContext('2d') : null;
        if (dctx) {
          new Chart(dctx, {
            type: 'doughnut',
            data: {
              labels: ['Churned','Retained'],
              datasets: [{
                data: [__CHURNED__, __RETAINED__],
                backgroundColor: ['#fb923c','#06b6d4']
              }]
            },
            options: {
              cutout: '70%',
              plugins: { legend: { position:'bottom', labels: { color:'#9aa3b2' } } }
            }
          });
        }

      </script>
    </body>
    </html>
    """

    html = HTML_TEMPLATE.replace("__VAL_MEDIAN__", f"{val_median:,}") \
                        .replace("__MONTHLY_TOTAL__", f"{monthly_total:,}") \
                        .replace("__CHURN_PCT__", f"{churn_pct:.1f}") \
                        .replace("__CHURNED__", str(churned)) \
                        .replace("__RETAINED__", str(retained)) \
                        .replace("__JS_LABELS__", js_labels) \
                        .replace("__JS_VALUES__", js_values) \
                        .replace("__JS_COMPS__", js_comps) \
                        .replace("__NUM_COMPS__", str(num_comps))

    components.html(html, height=920, scrolling=True)


def render_hybrid_component(hist):
    """
    Option 2 — Hybrid: Render just the KPI + small chart area in a components.html block,
    while letting Streamlit keep forms and interactivity (recommended quick win).
    """
    if hist is None or hist.empty:
        labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        values = [100, 120, 150, 170, 190, 210]
        val_median = 20000000
        monthly_total = 1200000
    else:
        try:
            tmp = hist.copy()
            tmp["month"] = tmp["Date"].dt.to_period("M").dt.to_timestamp()
            monthly = tmp.groupby("month")["Price"].sum().reset_index().sort_values("month").tail(6)
            labels = [d.strftime("%b") for d in monthly["month"].tolist()]
            values = [int(v) for v in monthly["Price"].tolist()]
        except Exception:
            labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
            values = [100, 120, 150, 170, 190, 210]
        try:
            val_median = int(hist["Price"].median())
        except Exception:
            val_median = 0
        try:
            cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days=30))
            monthly_total = int(hist[hist["Date"] >= cutoff]["Price"].sum())
        except Exception:
            monthly_total = sum(values) if values else 0

    js_labels = json.dumps(labels)
    js_values = json.dumps(values)
    HYBRID_TEMPLATE = """
    <div style="font-family:Inter, sans-serif; color:#e6eef6; background:transparent;">
      <div style="display:flex; gap:12px; margin-bottom:8px;">
        <div style="background:linear-gradient(90deg,#2f6bff,#23c7b8); padding:10px 12px; border-radius:10px; font-weight:700;">HC</div>
        <div>
          <div style="font-weight:700; font-size:16px;">Capstone</div>
          <div style="color:#9aa3b2; font-size:12px;">Real Estate • E-Commerce • Churn</div>
        </div>
      </div>

      <div style="display:flex; gap:12px; margin-top:8px;">
        <div style="background:rgba(255,255,255,0.03); padding:12px; border-radius:10px; width:50%;">
          <div style="color:#9aa3b2; font-size:12px;">Valuation Estimate</div>
          <div style="font-weight:800; font-size:20px; margin-top:6px;">₹__VAL_MEDIAN__</div>
          <div style="color:#7ee787; font-size:12px; margin-top:6px;">+5.2% vs last month</div>
        </div>

        <div style="background:rgba(255,255,255,0.03); padding:12px; border-radius:10px; width:50%;">
          <div style="color:#9aa3b2; font-size:12px;">Monthly Sales</div>
          <div style="font-weight:800; font-size:20px; margin-top:6px;">₹__MONTHLY_TOTAL__</div>
          <div style="color:#7ee787; font-size:12px; margin-top:6px;">+2.8% vs last month</div>
        </div>
      </div>

      <div style="margin-top:12px; background:rgba(255,255,255,0.02); padding:10px; border-radius:8px;">
        <canvas id="miniChart" width="400" height="140"></canvas>
      </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
      const labels = __JS_LABELS__;
      const values = __JS_VALUES__;
      const ctx = document.getElementById('miniChart') && document.getElementById('miniChart').getContext ? document.getElementById('miniChart').getContext('2d') : null;
      if (ctx) {
        new Chart(ctx, {
          type: 'line',
          data: {
            labels: labels,
            datasets: [{
              label: 'Sales',
              data: values,
              borderColor: 'rgba(47,107,255,0.95)',
              backgroundColor: 'rgba(47,107,255,0.08)',
              fill: true,
              tension: 0.3,
              pointRadius: 3
            }]
          },
          options: {
            plugins: { legend: { display: false } },
            scales: {
              x: { ticks: { color:'#9aa3b2' }, grid: { display: false } },
              y: { ticks: { color:'#9aa3b2' }, grid: { color:'rgba(255,255,255,0.02)' } }
            }
          }
        });
      }
    </script>
    """

    html = HYBRID_TEMPLATE.replace("__VAL_MEDIAN__", f"{val_median:,}") \
                          .replace("__MONTHLY_TOTAL__", f"{monthly_total:,}") \
                          .replace("__JS_LABELS__", js_labels) \
                          .replace("__JS_VALUES__", js_values)

    components.html(html, height=260, scrolling=False)

# ---------- END: Custom HTML / Hybrid visual helpers ----------

# -------------------- Sidebar (nav + theme) --------------------
with st.sidebar:
    st.markdown("""
    <div style="display:flex; gap:12px; align-items:center; margin-bottom:6px;">
      <div style="width:52px; height:52px; border-radius:12px; background: linear-gradient(90deg,#2f6bff,#23c7b8); display:flex; align-items:center; justify-content:center; color:white; font-weight:800; font-family: 'Poppins', sans-serif;">
        HC
      </div>
      <div>
        <div class="brand-title">Capstone</div>
        <div class="brand-sub">Real Estate • E-Commerce • Churn</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio("Navigate", options=["Dashboard", "Valuation", "Sales", "Churn", "Settings"], index=0)

    st.markdown("---")
    chosen = st.selectbox("Theme (preview)", options=list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state.ui_theme))
    if chosen != st.session_state.ui_theme:
        st.session_state.ui_theme = chosen
        inject_theme_css(st.session_state.ui_theme)
        st.experimental_rerun()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("© 2026 Capstone Analytics Suite", unsafe_allow_html=True)

# -------------------- Animated KPI JS (component) --------------------
ANIM_JS = """
<script>
function animateCount(id, endValue, duration=1200) {
  const el = document.getElementById(id);
  if(!el) return;
  const start = 0;
  const range = endValue - start;
  const startTime = performance.now();
  function step(now) {
    const progress = Math.min((now - startTime) / duration, 1);
    const eased = progress < 0.5 ? 2*progress*progress : -1 + (4 - 2*progress)*progress;
    const current = Math.round(start + eased * range);
    el.innerText = current.toLocaleString();
    if(progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}
function animateDecimal(id, endValue, duration=1200, decimals=1) {
  const el = document.getElementById(id);
  if(!el) return;
  const start = 0;
  const range = endValue - start;
  const startTime = performance.now();
  function step(now) {
    const progress = Math.min((now - startTime) / duration, 1);
    const eased = progress < 0.5 ? 2*progress*progress : -1 + (4 - 2*progress)*progress;
    const current = start + eased * range;
    el.innerText = current.toFixed(decimals) + "%";
    if(progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}
document.addEventListener("DOMContentLoaded", function() {
  document.querySelectorAll("[data-kpi-id]").forEach(function(node, idx) {
    const id = node.getAttribute("data-kpi-id");
    const val = parseFloat(node.getAttribute("data-kpi-value")) || 0;
    const isPercent = node.getAttribute("data-kpi-percent") === "true";
    setTimeout(()=> {
      if(isPercent) animateDecimal(id, val, 1200, 1);
      else animateCount(id, val, 1200);
    }, 120 * idx);
  });
});
</script>
"""
components.html(ANIM_JS, height=0)

# -------------------- Top hero --------------------
st.markdown("""
<div>
  <h1>Welcome back! 🚀</h1>
  <div class="hero-lead">Here’s your latest real estate comps, sales & churn insights.</div>
</div>
""", unsafe_allow_html=True)

# -------------------- DASHBOARD RENDERER --------------------
def render_dashboard():
    hist = load_historical_sales_cached()

    # compute safe KPI values
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

    val_num = int(valuation_est) if valuation_est else 20000000
    sales_num = int(monthly_sales) if monthly_sales else 1200000
    churn_pct = 8.5
    tasks_num = 4

    # KPI grid container
    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)

    def uid(prefix="kpi"):
        return prefix + "_" + str(random.randint(10000, 99999))

    id_val = uid("val")
    id_sales = uid("sales")
    id_churn = uid("churn")
    id_tasks = uid("tasks")

    # KPI items (safe markup)
    st.markdown(f'''
    <div class="kpi card">
      <div style="display:flex; align-items:center;">
        <div class="tile" style="background: linear-gradient(135deg, {THEMES[st.session_state.ui_theme]['accent_from']}, {THEMES[st.session_state.ui_theme]['accent_to']}); box-shadow: 0 0 20px rgba(47,107,255,0.12);">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M3 11.5L12 4l9 7.5" stroke="white" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </div>
        <div style="flex:1; margin-left:12px;">
          <div class="title">Valuation Estimate</div>
          <div class="value">₹<span id="{id_val}" data-kpi-id="{id_val}" data-kpi-value="{val_num}">0</span></div>
          <div class="delta" style="color:#7ee787;">+5.2% vs last month</div>
        </div>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown(f'''
    <div class="kpi card">
      <div style="display:flex; align-items:center;">
        <div class="tile" style="background: linear-gradient(135deg, {THEMES[st.session_state.ui_theme]['accent_to']}, {THEMES[st.session_state.ui_theme]['accent_from']}); box-shadow: 0 0 20px rgba(35,199,184,0.12);">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M3 3h2l1 9h11l3-6H6.5" stroke="white" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </div>
        <div style="flex:1; margin-left:12px;">
          <div class="title">Monthly Sales</div>
          <div class="value">₹<span id="{id_sales}" data-kpi-id="{id_sales}" data-kpi-value="{sales_num}">0</span></div>
          <div class="delta" style="color:#7ee787;">+2.8% vs last month</div>
        </div>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown(f'''
    <div class="kpi card" style="background: linear-gradient(90deg, rgba(245,158,11,0.06), rgba(255,255,255,0.00));">
      <div style="display:flex; align-items:center;">
        <div class="tile" style="background: linear-gradient(135deg,#f59e0b,#fb923c); box-shadow: 0 0 20px rgba(245,158,11,0.12);">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M12 9v3" stroke="white" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </div>
        <div style="flex:1; margin-left:12px;">
          <div class="title">Churn Risk</div>
          <div class="value"><span id="{id_churn}" data-kpi-id="{id_churn}" data-kpi-value="{churn_pct}" data-kpi-percent="true">0%</span></div>
          <div class="delta" style="color:#ffb86b;">-0.4% vs last month</div>
        </div>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown(f'''
    <div class="kpi card" style="background: linear-gradient(90deg, rgba(109,40,217,0.06), rgba(255,255,255,0.00));">
      <div style="display:flex; align-items:center;">
        <div class="tile" style="background: linear-gradient(135deg,#6d28d9,#8b5cf6); box-shadow: 0 0 20px rgba(109,40,217,0.12);">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M8 6h11" stroke="white" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </div>
        <div style="flex:1; margin-left:12px;">
          <div class="title">Open Tasks</div>
          <div class="value"><span id="{id_tasks}" data-kpi-id="{id_tasks}" data-kpi-value="{tasks_num}">0</span></div>
          <div class="delta" style="color:#a0b0c6;">2 new this week</div>
        </div>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Main content: left comps + right charts
    left, right = st.columns([2, 1], gap="large")
    with left:
        st.markdown('<div class="card"><div style="font-size:18px;font-weight:600;margin-bottom:12px;">📍 Latest Comparable Sales</div>', unsafe_allow_html=True)
        if not hist.empty and {"Date", "Price", "Area"}.issubset(set(hist.columns)):
            recent = hist.sort_values("Date", ascending=False).head(6)
            if {"lat", "lon"}.issubset(set(recent.columns)) and FOLIUM_AVAILABLE:
                try:
                    m = folium.Map(location=[recent.iloc[0]["lat"], recent.iloc[0]["lon"]], zoom_start=13, tiles="CartoDB.Dark_Matter")
                    mc = MarkerCluster()
                    for _, r in recent.iterrows():
                        try:
                            folium.CircleMarker(location=[r["lat"], r["lon"]],
                                                radius=6,
                                                tooltip=f'₹{int(r["Price"]):,} — {int(r.get("Area",0))} sqft',
                                                color="#23c7b8",
                                                fill=True,
                                                fill_opacity=0.9).add_to(mc)
                        except Exception:
                            pass
                    mc.add_to(m)
                    st_folium(m, width="100%", height=240)
                except Exception as e:
                    logger.exception("Folium render failed: %s", e)
            # show top 3 in a clean list
            for i, row in recent.head(3).iterrows():
                loc = row.get("Location", f"{row.get('Address', 'Comp')}")
                area = int(row.get("Area", 0))
                price = int(row.get("Price", 0))
                st.markdown(f"**{loc}** — {area:,} sqft — ₹{price:,}")
            st.markdown("---")
            st.dataframe(recent[["Date", "Price", "Area"]].head(6))
        else:
            st.info("No historical comps found. Upload `data/processed/historical_sales.csv` for comps list.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card"><div style="font-size:18px;font-weight:600;margin-bottom:12px;">📈 Sales Growth Summary <span style="font-size:13px;color: '+ THEMES[st.session_state.ui_theme]["muted"] +';">+2.8%</span></div>', unsafe_allow_html=True)
        try:
            import plotly.express as px
            tpl = _plotly_dark_template()
            if not hist.empty and "Date" in hist.columns and "Price" in hist.columns:
                tmp = hist.copy()
                tmp["month"] = tmp["Date"].dt.to_period("M").dt.to_timestamp()
                monthly = tmp.groupby("month")["Price"].sum().reset_index().sort_values("month")
                fig = px.line(monthly, x="month", y="Price", markers=True, template=tpl)
                fig.update_traces(line=dict(width=3.5, color=THEMES[st.session_state.ui_theme]["accent_from"]),
                                  marker=dict(size=6))
                fig.update_layout(margin=dict(l=0, r=0, t=8, b=0), height=320, font=dict(color=THEMES[st.session_state.ui_theme]["title"]))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sales history available to show growth chart.")
        except Exception as e:
            logger.exception("Plotly chart failed: %s", e)
            st.info("Chart rendering failed.")
        st.markdown('</div>', unsafe_allow_html=True)

    # churn overview small card below right column
    left2, right2 = st.columns([2, 1], gap="large")
    with left2:
        pass  # reserved space for additional content
    with right2:
        st.markdown('<div class="card"><div style="font-size:18px;font-weight:600;margin-bottom:12px;">📉 Churn Overview</div><div style="font-size:22px;font-weight:700;">8.5%</div><div style="color:'+THEMES[st.session_state.ui_theme]["muted"]+'">Churned: 425 • Retained: 4,575</div></div>', unsafe_allow_html=True)

# -------------------- VALUATION PAGE (logic preserved, visual improved) --------------------
def render_valuation():
    st.markdown("## 🏠 AI Property Valuation Engine")
    st.markdown('<div class="card"><div style="font-size:16px;font-weight:600;margin-bottom:8px;">Global ML valuation + hyperlocal adjustment using recent comps.</div>', unsafe_allow_html=True)

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

            input_data = {"Area": area, "Bedrooms": bedrooms, "Bathrooms": bathrooms, "Age": age, "Property_Type": property_type}
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

            # Results (styled)
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
                        st.write(comps[["Date", "Price", "Area", "Bedrooms", "Bathrooms"]].head(10))
                    st.dataframe(comps.sort_values("Date", ascending=False).head(20))
                    csv = comps.to_csv(index=False)
                    st.download_button("Download comps CSV", csv, "comps.csv")
                else:
                    st.info("No comps to show.")
            with result_col2:
                st.subheader("Map")
                if FOLIUM_AVAILABLE and (lat is not None and lon is not None):
                    m = folium.Map(location=[lat, lon], zoom_start=14, tiles="CartoDB.Dark_Matter")
                    folium.Marker([lat, lon], tooltip="Target", icon=folium.Icon(color="lightred")).add_to(m)
                    if not comps.empty:
                        for _, r in comps.iterrows():
                            try:
                                folium.CircleMarker(location=[r["lat"], r["lon"]],
                                                    radius=6,
                                                    tooltip=f'₹{int(r["Price"]):,}',
                                                    color="#23c7b8",
                                                    fill=True,
                                                    fill_opacity=0.9).add_to(m)
                            except Exception:
                                pass
                    st_folium(m, width=350, height=450)
                else:
                    if lat is not None and lon is not None:
                        st.map(pd.DataFrame([{"lat": lat, "lon": lon}]))
                    else:
                        st.info("No location to show. Provide address for map and local comps.")

# -------------------- SALES PAGE (unchanged logic, clean layout) --------------------
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

# -------------------- CHURN PAGE (unchanged logic, clean layout) --------------------
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

# -------------------- SETTINGS --------------------
def render_settings():
    st.header("Settings")
    st.markdown("Use this page to configure API keys, theme defaults, or other admin settings.")
    st.markdown("**Current theme:** " + st.session_state.ui_theme)

# -------------------- Router --------------------
current_page = page  # from sidebar radio

if current_page == "Dashboard":

    visual_mode = st.sidebar.selectbox(
        "Visual Mode (test)",
        ["native", "full_html", "hybrid"],
        index=0
    )

    hist_for_vis = load_historical_sales_cached()

    if visual_mode == "full_html":
        render_custom_html_full(hist_for_vis)

    elif visual_mode == "hybrid":
        render_hybrid_component(hist_for_vis)
        st.markdown("---")
        render_dashboard()

    else:  # native
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
    st.error("Page not found")

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Capstone Analytics Suite • UI & UX upgraded — core ML logic unchanged")