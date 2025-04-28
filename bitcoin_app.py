import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go   # For dynamic charts
import datetime
from plotly.subplots import make_subplots

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
timeframe = st.selectbox("Candle Timeframe:", ["Daily", "Weekly", "Monthly"])
indicators = st.multiselect("Select Indicators:", [
    "SMA_9", "SMA_20", "SMA_50", "SMA_200",
    "EMA_9", "EMA_20", "EMA_50", "EMA_200",
    "Bollinger Bands", "RSI", "MACD"
])

# === 2️⃣ Filter Data ===
prefix = asset_options[asset_choice]
price_cols = [f'Open_{prefix}', f'High_{prefix}', f'Low_{prefix}', f'Close_{prefix}']
volume_col = f'Volume_{prefix}' if f'Volume_{prefix}' in master_df_dashboard.columns else None

df_plot = master_df_dashboard[price_cols].copy()
if volume_col:
    df_plot['Volume'] = master_df_dashboard[volume_col]

# === 3️⃣ Resample Timeframe Safely ===
if timeframe in ["Weekly", "Monthly"]:
    resample_rule = 'W' if timeframe == "Weekly" else 'M'
    agg_dict = {
        price_cols[0]: 'first',
        price_cols[1]: 'max',
        price_cols[2]: 'min',
        price_cols[3]: 'last'
    }
    if volume_col:
        agg_dict['Volume'] = 'sum'
    df_plot = df_plot.resample(resample_rule).agg(agg_dict)

# === 4️⃣ Setup Subplots ===
rows = 1
if "RSI" in indicators:
    rows += 1
if "MACD" in indicators:
    rows += 1

fig = make_subplots(rows=rows, shared_xaxes=True, vertical_spacing=0.02,
                    row_heights=[0.6] + [0.2]*(rows-1))

current_row = 1

# === 5️⃣ Main Price Chart ===
if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot[price_cols[0]],
        high=df_plot[price_cols[1]],
        low=df_plot[price_cols[2]],
        close=df_plot[price_cols[3]],
        name="Price"
    ), row=current_row, col=1)
else:
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot[price_cols[3]],
        mode='lines',
        name='Close Price'
    ), row=current_row, col=1)

# === 6️⃣ Overlay Indicators ===
for ind in indicators:
    if ind.startswith("SMA") or ind.startswith("EMA"):
        col_name = f"{ind}_Close_{prefix}"
        if col_name in master_df_dashboard.columns:
            fig.add_trace(go.Scatter(x=master_df_dashboard.index, y=master_df_dashboard[col_name], name=ind), row=1, col=1)
    if ind == "Bollinger Bands":
        upper = f'Upper_Band_Close_{prefix}'
        lower = f'Lower_Band_Close_{prefix}'
        if upper in master_df_dashboard.columns and lower in master_df_dashboard.columns:
            fig.add_trace(go.Scatter(x=master_df_dashboard.index, y=master_df_dashboard[upper], name='Upper Band', line=dict(dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=master_df_dashboard.index, y=master_df_dashboard[lower], name='Lower Band', line=dict(dash='dot')), row=1, col=1)

# === 7️⃣ RSI Subplot ===
if "RSI" in indicators:
    current_row += 1
    rsi_col = f'RSI_Close_{prefix}'
    if rsi_col in master_df_dashboard.columns:
        fig.add_trace(go.Scatter(x=master_df_dashboard.index, y=master_df_dashboard[rsi_col], name="RSI", line=dict(color='orange')), row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", row=current_row, col=1, range=[0, 100])

# === 8️⃣ MACD Subplot with BTC Fix ===
if "MACD" in indicators:
    current_row += 1
    if prefix == "BTC-USD":
        macd_col = 'MACD_BTC'
        signal_col = 'Signal_Line_BTC'
    else:
        macd_col = f'MACD_{prefix}'
        signal_col = f'Signal_Line_{prefix}'

    if macd_col in master_df_dashboard.columns and signal_col in master_df_dashboard.columns:
        fig.add_trace(go.Scatter(x=master_df_dashboard.index, y=master_df_dashboard[macd_col], name="MACD", line=dict(color='purple')), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=master_df_dashboard.index, y=master_df_dashboard[signal_col], name="Signal", line=dict(color='gray', dash='dot')), row=current_row, col=1)
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)

# === 9️⃣ Layout Settings ===
fig.update_layout(
    title=f"{asset_choice} Price Chart",
    height=300 + rows * 200,
    xaxis_rangeslider_visible=False,
    showlegend=True
)

# === 🔟 Display Chart ===
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

st.markdown("---")

# =========================================
# 📊 Spot Bitcoin ETF Flows Visualization
# =========================================
st.subheader("📊 Spot Bitcoin ETF Inflows/Outflows")

# --- Make sure the index is datetime ---
etf_flow.index = pd.to_datetime(etf_flow.index)

# --- Daily Net Flow from 'Total' column ---
net_flow = etf_flow['Total']

# --- Bar Chart of Daily Net Flows ---
fig_etf = go.Figure()

fig_etf.add_trace(go.Bar(
    x=net_flow.index,
    y=net_flow,
    marker_color=['green' if val >= 0 else 'red' for val in net_flow],
    name='Daily Net Flow'
))

fig_etf.update_layout(
    title="Daily Net Flows of Spot BTC ETFs (All Funds)",
    xaxis_title="Date",
    yaxis_title="Net Flow (USD)",
    showlegend=False,
    height=400
)

st.plotly_chart(fig_etf, use_container_width=True)

# --- Cumulative Flow Line Chart ---
cumulative_flow = net_flow.cumsum()
st.line_chart(cumulative_flow, height=300)

# --- Latest Stats ---
latest_val = net_flow.iloc[-1]
total_flow = cumulative_flow.iloc[-1]

st.info(f"**Latest Net Flow:** {'+' if latest_val >=0 else ''}{latest_val:,.0f} USD")
st.success(f"**Total Cumulative Flow:** {total_flow:,.0f} USD")

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
