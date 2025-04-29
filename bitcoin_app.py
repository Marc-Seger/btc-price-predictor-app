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
etf_flow = pd.read_csv('data/ETF_Flow_Cleaned.csv', parse_dates=['Date'], index_col='Date')
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

# =========================================
# 🤖 Bitcoin Price Predictor (Coming Soon)
# =========================================
st.subheader("🤖 Bitcoin Price Predictor")

st.markdown("Select a future date to predict Bitcoin's price. This feature is under development.")

col1, col2 = st.columns([3,1])

with col1:
    future_date = st.date_input("Select Prediction Date", value=pd.Timestamp.today() + pd.Timedelta(days=7))

with col2:
    st.button("🔒 Predict", disabled=True)

st.warning("🚧 This feature is a Work In Progress. Stay tuned!")

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

# --- Ensure Clean Data ---
etf_flow = etf_flow.copy()
etf_flow.index = pd.to_datetime(etf_flow.index, errors='coerce')
net_flow = pd.to_numeric(etf_flow['Total'], errors='coerce')
cumulative_flow = net_flow.cumsum()

# ====================================
# 📅 Bar Chart: Daily Net Flows (US$M)
# ====================================
fig_etf = go.Figure()

fig_etf.add_trace(go.Bar(
    x=net_flow.index,
    y=net_flow,
    marker_color=['green' if val >= 0 else 'red' for val in net_flow],
    hovertemplate='%{y:.1f} M USD on %{x|%b %d, %Y}<extra></extra>',
    name='Daily Net Flow'
))

fig_etf.update_layout(
    title="Daily Net Flows of Spot BTC ETFs",
    xaxis_title="Date",
    yaxis_title="Net Flow (US$M)",
    showlegend=False,
    height=400,
    xaxis=dict(
        rangeslider=dict(visible=False),
        type="date"
    ),
    template="plotly_dark"
)

st.plotly_chart(fig_etf, use_container_width=True)

# =====================================================
# 📈 Cumulative Flow Line Chart with Dynamic Dropdown
# =====================================================

# --- Convert Cumulative Flow to Billions ---
cumulative_flow_billion = cumulative_flow / 1000  # Convert M to B

# Dropdown for dynamic range selection
range_option = st.selectbox(
    "Select Time Range:",
    options=["7d", "14d", "1m", "3m", "6m", "All"],
    index=2  # Default to "1m"
)

# Calculate date range based on selection
if range_option != "All":
    days_map = {"7d": 7, "14d": 14, "1m": 30, "3m": 90, "6m": 180}
    days = days_map[range_option]
    start_date = cumulative_flow.index.max() - pd.Timedelta(days=days)
    filtered_cumul = cumulative_flow_billion[cumulative_flow.index >= start_date]
else:
    filtered_cumul = cumulative_flow_billion

# Plotly Figure
fig_cumulative = go.Figure()

fig_cumulative.add_trace(go.Scatter(
    x=filtered_cumul.index,
    y=filtered_cumul.values,
    mode='lines+markers',
    name='Cumulative Flow',
    line=dict(color='lightskyblue')
))

fig_cumulative.update_layout(
    title="Cumulative Spot BTC ETF Flows",
    xaxis_title="Date",
    yaxis_title="Cumulative Flow (US$B)",
    template="plotly_dark",
    yaxis=dict(autorange=True)  # Dynamically adjust Y-axis
)

st.plotly_chart(fig_cumulative, use_container_width=True)

# ======================
# 📢 Latest Stats Summary
# ======================
latest_val = net_flow.iloc[-1]
total_flow_billion = cumulative_flow_billion.iloc[-1]

st.info(f"**Latest Net Flow:** {'+' if latest_val >=0 else ''}{latest_val:,.1f} M USD")
st.success(f"**Total Cumulative Flow:** {total_flow_billion:,.2f} B USD")

st.markdown("---")

# =========================================
# 🚨 Signals & Insights
# =========================================
st.subheader("🚨 Signals & Insights")

# --- Overall Market Sentiment Placeholder ---
overall_sentiment_placeholder = st.empty()

# --- Define Asset Prefixes ---
asset_prefixes = {
    "BTC": "BTC-USD",
    "SP500": "SP500",
    "NASDAQ": "NASDAQ",
    "GOLD": "Gold",
    "DXY": "DXY"
}

# --- Helper to Extract Asset Data ---
def get_asset_data(asset_key):
    prefix = asset_prefixes[asset_key]
    cols = [col for col in master_df_dashboard.columns if f"_{prefix}" in col or col.startswith(f"{prefix}")]
    return master_df_dashboard[cols].copy()

# --- Define Priority Order (OUTSIDE the loop) ---
priority_order = [
    "🟢 Golden Cross",
    "🔴 Death Cross",
    "📈 Above Upper Bollinger Band",
    "📉 Below Lower Bollinger Band",
    "⚡ Bullish Momentum",
    "⚡ Bearish Momentum",
    "📈 Stochastic Overbought",
    "📉 Stochastic Oversold",
    "📈 RSI Overbought",
    "📉 RSI Oversold",
    "📈 Price Above VWAP",
    "🚀 High Volume Breakout"
]

# --- Generate Signal Data from Precomputed Columns ---
summary_data = {"Asset": [], "Signal Summary": []}
detailed_data = []

for asset_key, prefix in asset_prefixes.items():
    df = get_asset_data(asset_key)
    signals = []

    # Moving Averages (Golden/Death Cross)
    if f'Golden_Cross_{asset_key}' in df.columns and df[f'Golden_Cross_{asset_key}'].iloc[-1] == 1:
        signals.append("🟢 Golden Cross")
    elif f'Death_Cross_{asset_key}' in df.columns and df[f'Death_Cross_{asset_key}'].iloc[-1] == 1:
        signals.append("🔴 Death Cross")

    # Bollinger Bands
    if f'Bollinger_Upper_Break_{asset_key}' in df.columns and df[f'Bollinger_Upper_Break_{asset_key}'].iloc[-1] == 1:
        signals.append("📈 Above Upper Bollinger Band")
    elif f'Bollinger_Lower_Break_{asset_key}' in df.columns and df[f'Bollinger_Lower_Break_{asset_key}'].iloc[-1] == 1:
        signals.append("📉 Below Lower Bollinger Band")

    # MACD Momentum
    if f'MACD_Above_Signal_{asset_key}' in df.columns:
        macd_signal = "⚡ Bullish Momentum" if df[f'MACD_Above_Signal_{asset_key}'].iloc[-1] == 1 else "⚡ Bearish Momentum"
        signals.append(macd_signal)

    # Stochastic Oscillator
    if f'Stoch_Overbought_{asset_key}' in df.columns and df[f'Stoch_Overbought_{asset_key}'].iloc[-1] == 1:
        signals.append("📈 Stochastic Overbought")
    elif f'Stoch_Oversold_{asset_key}' in df.columns and df[f'Stoch_Oversold_{asset_key}'].iloc[-1] == 1:
        signals.append("📉 Stochastic Oversold")

    # RSI
    if f'RSI_Overbought_{asset_key}' in df.columns and df[f'RSI_Overbought_{asset_key}'].iloc[-1] == 1:
        signals.append("📈 RSI Overbought")
    elif f'RSI_Oversold_{asset_key}' in df.columns and df[f'RSI_Oversold_{asset_key}'].iloc[-1] == 1:
        signals.append("📉 RSI Oversold")

    # VWAP
    if f'Price_Above_VWAP_{asset_key}' in df.columns and df[f'Price_Above_VWAP_{asset_key}'].iloc[-1] == 1:
        signals.append("📈 Price Above VWAP")

    # Volume Breakout
    if f'High_Volume_{asset_key}' in df.columns and df[f'High_Volume_{asset_key}'].iloc[-1] == 1:
        signals.append("🚀 High Volume Breakout")

    # --- Sort signals by priority ---
    signals_sorted = sorted(signals, key=lambda x: priority_order.index(x) if x in priority_order else 999)

    # --- Fill summary_data ---
    if not signals_sorted:
        summary_data["Signal Summary"].append("⚠️ No significant signals")
    else:
        summary_data["Signal Summary"].append("\n".join(signals_sorted))
    summary_data["Asset"].append(asset_key)

    # --- Fill detailed_data ---
    today_date = master_df_dashboard.index[-1].strftime('%Y-%m-%d')
    for signal in signals_sorted:
        detailed_data.append([asset_key, signal, "Active", today_date])

# --- Quick Summary Table ---
st.markdown("### 📊 Technical Signals Summary")
st.dataframe(pd.DataFrame(summary_data), hide_index=True)

# --- Interactive Detailed Signals ---
st.markdown("### Explore Detailed Signals")

asset_options = ["None", "All"] + list(asset_prefixes.keys())
signal_types = ["None", "Moving Averages", "Bollinger Bands", "Momentum"]

selected_asset = st.selectbox("Select Asset", asset_options)

detailed_df = pd.DataFrame(detailed_data, columns=["Asset", "Signal Type", "Status", "Date"])

if selected_asset == "None":
    filtered_df = pd.DataFrame(columns=["Asset", "Signal Type", "Status", "Date"])
elif selected_asset == "All":
    filtered_df = detailed_df
else:
    filtered_df = detailed_df[detailed_df["Asset"] == selected_asset]

st.dataframe(filtered_df, hide_index=True)

# --- Volume Breakout Alerts ---
st.markdown("### 🚨 Volume Breakout Alerts (>150% Avg Volume)")

volume_alerts = []
for asset_key, prefix in asset_prefixes.items():
    # Special case for Gold volume to use the capped version
    if prefix == "Gold":
        vol_col = 'Volume_Gold_Capped'
    else:
        vol_col = f'Volume_{prefix}'
        
    if vol_col in master_df_dashboard.columns:
        df = get_asset_data(asset_key)
        avg_vol = df[vol_col].rolling(window=20).mean().iloc[-1]
        current_vol = df[vol_col].iloc[-2]  # Use previous day's volume
        vol_date = df.index[-2].strftime('%Y-%m-%d')

        if current_vol > 1.5 * avg_vol:
            volume_alerts.append([asset_key, f"{current_vol:,.0f} ({vol_date})", f"{avg_vol:,.0f}", "⚡ Breakout"])
        else:
            volume_alerts.append([asset_key, f"{current_vol:,.0f} ({vol_date})", f"{avg_vol:,.0f}", "No Signal"])

vol_df = pd.DataFrame(volume_alerts, columns=["Asset", "Current Volume", "20D Avg Volume", "Signal"])
vol_df_sorted = vol_df.sort_values(by="Signal", ascending=True)
st.dataframe(vol_df_sorted, hide_index=True)

# --- Calculate Overall Market Sentiment (before any asset selection) ---
all_detailed_df = pd.DataFrame(detailed_data, columns=["Asset", "Signal Type", "Status", "Date"])

bullish_signals = all_detailed_df[all_detailed_df['Status'].str.contains("Bullish")]
bearish_signals = all_detailed_df[all_detailed_df['Status'].str.contains("Bearish")]

bullish_count = bullish_signals.shape[0]
bearish_count = bearish_signals.shape[0]

if bullish_count > bearish_count:
    bullish_assets = bullish_signals['Asset'].unique()
    overall_sentiment_placeholder.success(
        f"📢 Market Sentiment Based on Signals: **Bullish Bias** ({bullish_count} bullish signals: {', '.join(bullish_assets)})"
    )
elif bearish_count > bullish_count:
    bearish_assets = bearish_signals['Asset'].unique()
    overall_sentiment_placeholder.error(
        f"📢 Market Sentiment Based on Signals: **Bearish Bias** ({bearish_count} bearish signals: {', '.join(bearish_assets)})"
    )
else:
    overall_sentiment_placeholder.info("📢 Market Sentiment Based on Signals: **Neutral**")

st.markdown("---")

# =========================
# 😨 Sentiment Section
# =========================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 😨 Bitcoin Fear & Greed Index (14D)")
    st.line_chart(master_df_dashboard['BTC_index_value'].tail(14))

with col2:
    st.markdown("### 🔎 Google Trends: 'Bitcoin' (14D)")
    st.line_chart(google_trends['GT_index_bitcoin'].tail(14))

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
