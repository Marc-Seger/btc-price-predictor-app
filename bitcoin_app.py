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

# === 1️⃣ Chart Controls ===
asset_options = {
    "Bitcoin": "BTC-USD",
    "S&P 500": "SP500",
    "Nasdaq": "NASDAQ",
    "Gold": "Gold",
    "DXY": "DXY"
}

asset_choice = st.selectbox("Select Asset:", list(asset_options.keys()))
chart_type = st.radio("Chart Type:", ["Line Chart", "Candlestick"])
indicators = st.multiselect("Select Indicators:", ["SMA_20", "EMA_20", "Bollinger Bands"])

# === 2️⃣ Filter Data ===
prefix = asset_options[asset_choice]

price_cols = [f'Open_{prefix}', f'High_{prefix}', f'Low_{prefix}', f'Close_{prefix}']

# Check if asset has volume (DXY doesn't)
volume_col = f'Volume_{prefix}' if f'Volume_{prefix}' in master_df_dashboard.columns else None

df_plot = master_df_dashboard[price_cols].copy()
if volume_col:
    df_plot['Volume'] = master_df_dashboard[volume_col]

# === 3️⃣ Create Plotly Figure ===
fig = go.Figure()

if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot[price_cols[0]],
        high=df_plot[price_cols[1]],
        low=df_plot[price_cols[2]],
        close=df_plot[price_cols[3]],
        name="Price"
    ))
else:
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot[price_cols[3]],
        mode='lines',
        name='Close Price'
    ))

# === 4️⃣ Add Indicators ===
if "SMA_20" in indicators:
    sma_col = f'SMA_20_Close_{prefix}'
    if sma_col in master_df_dashboard.columns:
        fig.add_trace(go.Scatter(
            x=master_df_dashboard.index,
            y=master_df_dashboard[sma_col],
            mode='lines',
            name='SMA 20'
        ))

if "EMA_20" in indicators:
    ema_col = f'EMA_20_Close_{prefix}'
    if ema_col in master_df_dashboard.columns:
        fig.add_trace(go.Scatter(
            x=master_df_dashboard.index,
            y=master_df_dashboard[ema_col],
            mode='lines',
            name='EMA 20'
        ))

if "Bollinger Bands" in indicators:
    upper = f'Upper_Band_Close_{prefix}'
    lower = f'Lower_Band_Close_{prefix}'
    if upper in master_df_dashboard.columns and lower in master_df_dashboard.columns:
        fig.add_trace(go.Scatter(x=master_df_dashboard.index, y=master_df_dashboard[upper], name='Upper Band', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=master_df_dashboard.index, y=master_df_dashboard[lower], name='Lower Band', line=dict(dash='dot')))

# Layout tweaks
fig.update_layout(title=f"{asset_choice} Price Chart", xaxis_title="Date", yaxis_title="Price", height=600)

# === 5️⃣ Display Chart ===
st.plotly_chart(fig, use_container_width=True)

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
