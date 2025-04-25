import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go   # For dynamic charts
import datetime

# === Load Data ===
# === Load master_df_dashboard ===
master_df_dashboard = pd.read_csv('data/master_df_dashboard.csv', index_col=0, parse_dates=True)

# === Load Google Trends ===
google_trends = pd.read_csv('data/multiTimeline.csv', skiprows=1)
google_trends.rename(columns={google_trends.columns[0]: 'Date', google_trends.columns[1]: 'GT_index_bitcoin'}, inplace=True)
google_trends['Date'] = pd.to_datetime(google_trends['Date'])
google_trends.set_index('Date', inplace=True)
google_trends.ffill(inplace=True)

# === Load ETF Flow ===
etf_flow = pd.read_csv('data/SPOT BTC ETF IN_OUT_FLOW.csv', parse_dates=['Date'], index_col='Date')
# --- Page Config ---
st.set_page_config(page_title="Bitcoin Market Dashboard", page_icon="📊", layout="wide")

# --- Header ---
st.title("📊 Bitcoin & Market Intelligence Dashboard")
st.markdown("An interactive dashboard to monitor Bitcoin, financial markets, and key indicators. *(Work in Progress)*")

# --- KPI Cards (Placeholders) ---
st.subheader("📈 Market Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("BTC Price", "$94,200", "+2.3%")
col2.metric("Fear & Greed Index", "72 (Greed)", "↑")
col3.metric("Last ETF Net Flow", "+$180M", "")
col4.metric("24h Volume Spike", "Yes", "🚨")

st.markdown("---")

# --- Main Chart Section ---
st.subheader("🗺️ Asset Chart")

# Chart Controls
asset = st.selectbox("Select Asset:", ["Bitcoin", "S&P 500", "Nasdaq", "Gold", "DXY"])
chart_type = st.radio("Chart Type:", ["Line Chart", "Candlestick"])
indicators = st.multiselect("Indicators:", ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands"])
timeframe = st.selectbox("Timeframe:", ["Daily", "Weekly", "Monthly"])

# Chart Placeholder
st.info("📍 Chart will be displayed here with selected options. (Coming Soon)")

st.markdown("---")

# --- Signals & Insights ---
st.subheader("🚨 Signals & Insights")

col_sig1, col_sig2 = st.columns(2)

with col_sig1:
    st.markdown("**📊 Volume Breakout Signals**")
    st.dataframe({"Asset": ["BTC", "Nasdaq"], "Signal": ["Breakout", "No Signal"]})

    st.markdown("**⚡ Momentum Flags**")
    st.dataframe({"Asset": ["BTC", "Gold"], "Momentum": ["Strong Up", "Weak Down"]})

with col_sig2:
    st.markdown("**😨 Fear & Greed Index (Last 7 Days)**")
    st.line_chart([60, 65, 70, 68, 72, 75, 72])  # Placeholder data

    st.markdown("**💰 Bitcoin ETF Inflows/Outflows**")
    st.bar_chart([100, -50, 200, -30, 180])  # Placeholder data

    st.markdown("**🔎 Google Trends: 'Bitcoin'**")
    st.line_chart([40, 45, 60, 55, 70, 65, 68])  # Placeholder data

st.markdown("---")

# --- Roadmap ---
with st.expander("🛠️ Project Roadmap & Notes"):
    st.markdown("""
    - [x] Dashboard layout & UI setup
    - [ ] Integrate real market data
    - [ ] Implement dynamic charts with Plotly
    - [ ] Add predictive models
    - [ ] Deploy full version
    """)

# --- Footer ---
st.markdown("---")
st.caption("Created by Marc Seger | [GitHub](https://github.com/Marc-Seger) | [Portfolio](https://marc-seger.github.io/portfolio)")
