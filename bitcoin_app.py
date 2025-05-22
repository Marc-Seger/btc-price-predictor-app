import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
from plotly.subplots import make_subplots
import base64

# === Page Config ===
st.set_page_config(
    page_title="Bitcoin Dashboard",
    page_icon="images/favicon.ico",
    layout="wide"
)

# === Load Data ===
master_df_dashboard = pd.read_csv('data/master_df_dashboard.csv', index_col=0, parse_dates=True)

google_trends = pd.read_csv('data/multiTimeline.csv', skiprows=1)
google_trends.rename(columns={google_trends.columns[0]: 'Date', google_trends.columns[1]: 'GT_index_bitcoin'}, inplace=True)
google_trends['Date'] = pd.to_datetime(google_trends['Date'])
google_trends.set_index('Date', inplace=True)
google_trends.ffill(inplace=True)

etf_flow = pd.read_csv('data/etf_flow_cleaned.csv', parse_dates=['Date'], index_col='Date')

# === Image Load Function ===
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

bitcoin_logo_base64 = get_base64_image("images/bitcoin_logo.png")

# === Header Section ===
last_updated_date = master_df_dashboard.index.max().strftime("%Y-%m-%d")

st.markdown(
    f"""
    <div style='display: flex; align-items: flex-start; gap: 18px; margin-bottom: 0;'>
        <img src="data:image/png;base64,{bitcoin_logo_base64}" width="64" style="margin-top: 4px;" />
        <div style='margin-top: 8px;'>
            <div style='font-size: 2.8rem; font-weight: 800; color: white; margin-bottom: 6px;'>
                Bitcoin & Market Intelligence Dashboard
            </div>
            <div style='font-size: 1.1rem; color: white; margin-bottom: 4px;'>
                An interactive dashboard to monitor Bitcoin, financial markets, and key indicators.
                <em>(Last updated: {last_updated_date})</em>
            </div>
            <div style='font-size: 1.1rem; color: white;'>
                🔮 Looking to forecast Bitcoin price? Try the
                <a href="https://bitcoin-predictor.streamlit.app/" target="_blank"
                   style="color: #48b5ff; text-decoration: none;">Bitcoin Price Predictor</a> app!
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

# === KPI Calculations ===
btc_series = master_df_dashboard['Close_BTC']
btc_price = btc_series.iloc[-1]

lookback = 2
while lookback <= len(btc_series) and btc_series.iloc[-lookback] == btc_price:
    lookback += 1
if lookback > len(btc_series):
    btc_change = 0
    btc_text = "No Change"
else:
    btc_prev = btc_series.iloc[-lookback]
    btc_change = ((btc_price - btc_prev) / btc_prev) * 100
    btc_text = f"{btc_change:+.1f}% vs {lookback-1}D"
btc_color = "green" if btc_change > 0 else "red" if btc_change < 0 else "gray"

fng_value = master_df_dashboard['Sentiment_BTC_index_value'].iloc[-1]
fng_label = master_df_dashboard['Sentiment_BTC_index_label'].iloc[-1]
fng_1d = master_df_dashboard['Sentiment_BTC_index_value'].iloc[-2]
fng_1d_change = fng_value - fng_1d
fng_1d_color = "green" if fng_1d_change > 0 else "red" if fng_1d_change < 0 else "gray"
fng_1d_text = f"{fng_1d_change:+.1f} vs 1D"
if len(master_df_dashboard) >= 8:
    fng_7d = master_df_dashboard['Sentiment_BTC_index_value'].iloc[-8]
    fng_7d_change = fng_value - fng_7d
    fng_7d_color = "green" if fng_7d_change > 0 else "red" if fng_7d_change < 0 else "gray"
    fng_7d_text = f"{fng_7d_change:+.1f} vs 7D"
else:
    fng_7d_text = "N/A vs 7D"
    fng_7d_color = "gray"

# --- Synchronized Volume Spike & % Change ---
vol_series = master_df_dashboard['Volume_BTC'].dropna()

# Use only completed daily data
latest_volume = vol_series.iloc[-2]   # Yesterday
previous_volume = vol_series.iloc[-3] # Day before

# Calculate % change
vol_change = ((latest_volume - previous_volume) / previous_volume) * 100 if previous_volume != 0 else 0
vol_change_text = f"{vol_change:+.1f}% vs prev day"

# Use spike flag from yesterday
volume_spike = master_df_dashboard.iloc[-2]['High_Volume_BTC']
spike_status = "Yes" if volume_spike else "No"
spike_color = "green" if volume_spike else "white"

# === KPI Cards Layout ===
col1, col2, col3, col4 = st.columns(4)

# --- BTC Price Card ---
col1.markdown(f"""
    <div style='
        background-color: #262730;
        padding: 16px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        text-align: center;
        height: 100%;
    '>
        <div style='font-weight:600; font-size:1.1rem;'>BTC Price</div>
        <div style='font-size:2rem; font-weight:700; margin:0.2rem 0;'>${btc_price:,.0f}</div>
        <div style='font-size:0.9rem; color:{btc_color}; font-weight:500'>{btc_text}</div>
    </div>
""", unsafe_allow_html=True)

# --- Fear & Greed Card ---
col2.markdown(f"""
    <div style='
        background-color: #262730;
        padding: 16px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        text-align: center;
        height: 100%;
    '>
        <div style='font-weight:600; font-size:1.1rem;'>Fear & Greed Index</div>
        <div style='font-size:2rem; font-weight:700; margin:0.2rem 0;'>{fng_value:.1f} ({fng_label})</div>
        <div style='font-size:0.9rem; font-weight:500; display:flex; justify-content:center; gap:12px;'>
            <span style='color:{fng_1d_color}'>{fng_1d_text}</span>
            <span style='color:{fng_7d_color}'>{fng_7d_text}</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- ETF Flow Card ---

# Define the ETF columns to aggregate
etf_columns = [
    "ETF_IBIT", "ETF_FBTC", "ETF_BITB", "ETF_ARKB", 
    "ETF_BTCO", "ETF_EZBC", "ETF_BRRR", "ETF_HODL", 
    "ETF_BTCW", "ETF_GBTC"
]

# Calculate the 'Total' column dynamically
etf_flow['Total'] = etf_flow[etf_columns].sum(axis=1)

# Ensure the calculated 'Total' column is numeric
etf_flow['Total'] = pd.to_numeric(etf_flow['Total'], errors='coerce')

# Get the latest and previous ETF net flow values
latest_etf_net_flow = etf_flow['Total'].iloc[-1]
prev_etf_net_flow = etf_flow['Total'].iloc[-2]

# Calculate % change
etf_flow_change = ((latest_etf_net_flow - prev_etf_net_flow) / abs(prev_etf_net_flow) * 100) if prev_etf_net_flow != 0 else 0
etf_flow_change_color = "green" if etf_flow_change > 0 else "red" if etf_flow_change < 0 else "gray"
etf_flow_change_text = f"{etf_flow_change:+.1f}% vs 1D"

col3.markdown(f"""
    <div style='
        background-color: #262730;
        padding: 16px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        text-align: center;
        height: 100%;
    '>
        <div style='font-weight:600; font-size:1.1rem;'>Last ETF Net Flow</div>
        <div style='font-size:2rem; font-weight:700; margin:0.2rem 0; color:white'>
            {latest_etf_net_flow:+,.0f}M USD
        </div>
        <div style='font-size:0.9rem; color:{etf_flow_change_color}; font-weight:500'>
            {etf_flow_change_text}
        </div>
    </div>
""", unsafe_allow_html=True)


# --- Volume Spike Card ---
col4.markdown(f"""
    <div style='
        background-color: #262730;
        padding: 16px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        text-align: center;
        height: 100%;
    '>
        <div style='font-weight:600; font-size:1.1rem;'>
            24h Volume Spike 
            <span title="Flags a spike if yesterday’s volume is in the top 10% (≥ 90th percentile) of recent data. Based on completed daily candles.">ℹ️</span>
        </div>
        <div style='font-size:2rem; font-weight:700; margin:0.2rem 0; color:{spike_color};'>
            {spike_status}
        </div>
        <div style='font-size:0.9rem; color:{"green" if vol_change > 0 else "red" if vol_change < 0 else "gray"}; font-weight:500'>
            {vol_change_text}
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Main Chart Section ---
st.subheader("Asset Chart")

# === 1️⃣ Chart Controls ===
asset_options = {
    "Bitcoin": "BTC",
    "S&P 500": "SP500",
    "Nasdaq": "NASDAQ",
    "Gold": "GOLD",
    "DXY": "DXY"
}

col1, col2, col3, col4 = st.columns([1.5, 1.8, 1.5, 2.2])

with col1:
    asset = st.selectbox("Select Asset", list(asset_options.keys()), key="asset_select")

with col2:
    timeframe = st.selectbox("Candle Timeframe", ["1H", "4H", "Daily", "Weekly"], key="timeframe_select")

with col3:
    indicators = st.multiselect(
        "Select Indicators",
        [
            # Moving Averages
            "SMA_20", "SMA_50", "SMA_200", 
            "EMA_9", "EMA_20", "EMA_50", "EMA_200", 
            # MACD (Daily and Weekly)
            "MACD_D",
            "MACD_W",
            # RSI
            "RSI",
            # Bollinger Bands
            "Bollinger Bands",
            # Stochastic Oscillator
            "Stochastic (%K, %D)",
            # VWAP
            "VWAP",
            # OBV
            "OBV"
        ],
        default=["SMA_50", "SMA_200"],  # Default selection
        key="indicator_select"
    )

# === 2️⃣ Filter Data ===
prefix = asset_options[asset]
price_cols = [f'Open_{prefix}', f'High_{prefix}', f'Low_{prefix}', f'Close_{prefix}']
volume_col = f'Volume_{prefix}' if f'Volume_{prefix}' in master_df_dashboard.columns else None

df_plot = master_df_dashboard[price_cols].copy()
if volume_col:
    df_plot['Volume'] = master_df_dashboard[volume_col]

# === 3️⃣ Resample Timeframe Safely ===
if timeframe == "Weekly":
    resample_rule = 'W'
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
# Determine the number of subplots based on selected indicators
rows = 1  # Main chart row
if "MACD_D" in indicators or "MACD_W" in indicators:
    rows += 1
if "RSI" in indicators:
    rows += 1
if "OBV" in indicators:
    rows += 1
if "Stochastic" in indicators:
    rows += 1

fig = make_subplots(rows=rows, shared_xaxes=True, vertical_spacing=0.03, 
                    row_heights=[0.5] + [0.12] * (rows - 1))

current_row = 1

# === 5️⃣ Main Price Chart ===
if asset == "Bitcoin":
    # Plot Candlestick for Bitcoin only
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot[price_cols[0]],
        high=df_plot[price_cols[1]],
        low=df_plot[price_cols[2]],
        close=df_plot[price_cols[3]],
        name="Price",
        increasing_line_color='green',
        decreasing_line_color='red'
    ), row=current_row, col=1)
else:
    # Plot Line Chart for all other assets
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot[price_cols[3]],  # Close Price for line chart
        mode='lines',
        name='Close Price',
        line=dict(color='white')
    ), row=current_row, col=1)


# === 6️⃣ Overlay Indicators (SMA, EMA, Bollinger Bands, VWAP) ===
for ind in indicators:
    # SMA/EMA
    if "SMA" in ind or "EMA" in ind:
        col_name = f"{ind}_Close_{prefix}"
        if col_name in master_df_dashboard.columns:
            fig.add_trace(go.Scatter(
                x=master_df_dashboard.index,
                y=master_df_dashboard[col_name],
                name=ind,
                line=dict(dash='dot')
            ), row=current_row, col=1)

    # Bollinger Bands
    if ind == "Bollinger Bands":
        upper = f'Upper_Band_Close_{prefix}'
        lower = f'Lower_Band_Close_{prefix}'
        if upper in master_df_dashboard.columns and lower in master_df_dashboard.columns:
            fig.add_trace(go.Scatter(
                x=master_df_dashboard.index,
                y=master_df_dashboard[upper],
                name='Upper Band',
                line=dict(dash='dash', color='blue')
            ), row=current_row, col=1)
            fig.add_trace(go.Scatter(
                x=master_df_dashboard.index,
                y=master_df_dashboard[lower],
                name='Lower Band',
                line=dict(dash='dash', color='red')
            ), row=current_row, col=1)

    # VWAP
    if ind == "VWAP":
        vwap_col = f'VWAP_30d_{prefix}'
        if vwap_col in master_df_dashboard.columns:
            fig.add_trace(go.Scatter(
                x=master_df_dashboard.index,
                y=master_df_dashboard[vwap_col],
                name='VWAP',
                line=dict(color='blue')
            ), row=current_row, col=1)

    # === 7️⃣ MACD (Daily & Weekly) Subplot ===
    if "MACD_D" in indicators or "MACD_W" in indicators:
        current_row += 1
        for macd_type in ["D", "W"]:
            if f"MACD_{macd_type}" in indicators:
                macd_col = f'MACD_{macd_type}_{prefix}'
                signal_col = f'Signal_Line_{macd_type}_{prefix}'
                hist_col = f'MACD_Histogram_{macd_type}_{prefix}'

                if macd_col in master_df_dashboard.columns and signal_col in master_df_dashboard.columns:
                    # MACD Line (Blue)
                    fig.add_trace(go.Scatter(
                        x=master_df_dashboard.index,
                        y=master_df_dashboard[macd_col],
                        name=f"MACD {macd_type}",
                        line=dict(color='#2962FF')  # Bright Blue
                    ), row=current_row, col=1)

                    # Signal Line (Orange)
                    fig.add_trace(go.Scatter(
                        x=master_df_dashboard.index,
                        y=master_df_dashboard[signal_col],
                        name=f"Signal {macd_type}",
                        line=dict(color='#FF6D00')  # Orange Dashed
                    ), row=current_row, col=1)

                    # Histogram (Teal or Red)
                    hist_vals = master_df_dashboard[hist_col]

                    fig.add_trace(go.Bar(
                        x=master_df_dashboard.index,
                        y=hist_vals,
                        name=f"Histogram {macd_type}",
                        marker_color=[
                            '#009688' if val >= 0 else '#F44336'
                            for val in hist_vals
                        ],
                        opacity=0.8
                    ), row=current_row, col=1)

                    #Force Y-axis to have a visible range
                    buffer = hist_vals.abs().max() * 1.2
                    fig.update_yaxes(title_text="MACD", row=current_row, col=1, range=[-buffer, buffer])

# === 8️⃣ RSI Subplot ===
if "RSI" in indicators:
    current_row += 1
    rsi_col = f'RSI_Close_{prefix}'
    if rsi_col in master_df_dashboard.columns:
        fig.add_trace(go.Scatter(
            x=master_df_dashboard.index,
            y=master_df_dashboard[rsi_col],
            name="RSI",
            line=dict(color='orange')
        ), row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", row=current_row, col=1, range=[0, 100])

# === 9️⃣ OBV Subplot ===
if "OBV" in indicators:
    current_row += 1
    obv_col = f'OBV_{prefix}'
    if obv_col in master_df_dashboard.columns:
        fig.add_trace(go.Scatter(
            x=master_df_dashboard.index,
            y=master_df_dashboard[obv_col],
            name="OBV",
            line=dict(color='green')
        ), row=current_row, col=1)

# === 🔟 Stochastic Subplot ===
if "Stochastic" in indicators:
    current_row += 1
    k_col = f'%K_{prefix}'
    d_col = f'%D_{prefix}'
    if k_col in master_df_dashboard.columns and d_col in master_df_dashboard.columns:
        fig.add_trace(go.Scatter(
            x=master_df_dashboard.index,
            y=master_df_dashboard[k_col],
            name="%K",
            line=dict(color='blue')
        ), row=current_row, col=1)
        fig.add_trace(go.Scatter(
            x=master_df_dashboard.index,
            y=master_df_dashboard[d_col],
            name="%D",
            line=dict(color='red')
        ), row=current_row, col=1)
        fig.update_yaxes(title_text="Stochastic", row=current_row, col=1, range=[0, 100])

# === 1️⃣1️⃣ Layout Settings ===
fig.update_layout(
    title=f"{asset} Price Chart with Indicators",
    height=400 + rows * 150,
    xaxis_rangeslider_visible=False,
    showlegend=True,
    template="plotly_dark"
)

# === 1️⃣2️⃣ Display Chart ===
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
st.markdown(
    f"<div style='background-color:#14532d; color:white; padding:10px; border-radius:10px;'>"
    f"📊 <strong>Total Cumulative Flow:</strong> {total_flow_billion:,.2f} B USD</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# =========================================
# 🚨 Signals & Insights
# =========================================
from datetime import timedelta

decay_params = {
    "Golden/Death Cross": {"decay_factor": 0.9845},
    "Weekly MACD": {"decay_factor": 0.995},
    "Daily MACD": {"decay_factor": 0.98},
    "RSI Signal": {"decay_factor": 0.97},
    "OBV": {"decay_factor": 1.0}  # No decay, daily signal
}

# === 🕒 Sentiment Timeframe Toggler ===
st.markdown("### 🕒 Sentiment Analysis Timeframe")
timeframe_option = st.selectbox(
    "Select Sentiment Analysis Period:",
    options=["30 Days", "90 Days", "180 Days"],
    index=0  # Default to "30 Days"
)

# Map timeframe selection to actual days
timeframe_map = {
    "30 Days": 30,
    "90 Days": 90,
    "180 Days": 180
}
selected_days = timeframe_map[timeframe_option]

# === 2️⃣ Extract Signals Dynamically ===

def extract_signals(df, asset_key, selected_days):
    signals = []
    prefix = asset_key

    start_date = datetime.datetime.now() - pd.Timedelta(days=selected_days)

    # Golden/Death Cross
    for cross_type, weight in [("Golden_Cross_Event", 3), ("Death_Cross_Event", -3)]:
        col_name = f"{cross_type}_{prefix}"
        if col_name in df.columns:
            for date in df[df[col_name] == 1].index:
                if date >= start_date:
                    # Assign a unified signal type for both cross types
                    signals.append({"type": "Golden/Death Cross", "date": date, "weight": weight, "asset": asset_key})

    # MACD - Daily and Weekly
    for macd_type, signal_label in [("D", "Daily MACD"), ("W", "Weekly MACD")]:
        macd_col = f"MACD_{macd_type}_{prefix}"
        if macd_col in df.columns:
            for date, value in df[macd_col].dropna().items():
                if date >= start_date:
                    weight = 2 if value > 0 else -2
                    signals.append({"type": signal_label, "date": date, "weight": weight, "asset": asset_key})

    # RSI - Overbought/Oversold
    rsi_col = f"RSI_Close_{prefix}"
    if rsi_col in df.columns:
        for date, value in df[rsi_col].dropna().items():
            if date >= start_date:
                if value >= 70:
                    signals.append({"type": "RSI Signal", "date": date, "weight": -1, "asset": asset_key})
                elif value <= 30:
                    signals.append({"type": "RSI Signal", "date": date, "weight": 1, "asset": asset_key})

    # OBV - Daily Signal
    obv_col = f'OBV_{prefix}'
    if obv_col in df.columns:
        for date, value in df[obv_col].diff().dropna().items():
            if date >= start_date:
                weight = 1 if value > 0 else -1
                signals.append({"type": "OBV", "date": date, "weight": weight, "asset": asset_key})

    return signals


# Collect signals for each asset
all_signals = []
for asset_key in ["BTC", "SP500", "NASDAQ", "GOLD", "DXY"]:
    asset_df = master_df_dashboard[[col for col in master_df_dashboard.columns if asset_key in col]]
    asset_signals = extract_signals(asset_df, asset_key, selected_days)
    all_signals.extend(asset_signals)

# === 3️⃣ Compute Sentiment Scores ===

def calculate_signal_weight(signal_date, signal_type, decay_type="multi_day"):
    age = (datetime.datetime.now() - signal_date).days
    decay_factor = decay_params[signal_type]["decay_factor"]

    if decay_type == "multi_day":
        return max(0, decay_factor ** age)
    elif decay_type == "daily":
        return 1  # Daily signals have a constant weight (OBV)
    return 0

def compute_mean_score(signals, decay_type):
    if not signals:
        return 0
    
    total_weight = 0
    total_days = len(signals)

    for signal in signals:
        weight = signal["weight"]
        date = signal["date"]
        signal_type = signal["type"]
        decay_weight = calculate_signal_weight(date, signal_type, decay_type)
        total_weight += weight * decay_weight
    
    return total_weight / total_days if total_days > 0 else 0

def compute_sentiment_scores(signals):
    # Separate signals by type
    multi_day_signals = [s for s in signals if s["type"] != "OBV"]
    daily_signals = [s for s in signals if s["type"] == "OBV"]

    # Compute mean scores
    mean_multi_day = compute_mean_score(multi_day_signals, "multi_day")
    mean_daily = compute_mean_score(daily_signals, "daily")

    return mean_multi_day, mean_daily

# Calculate sentiment scores
btc_signals = [s for s in all_signals if s["asset"] == "BTC"]
market_signals = [s for s in all_signals if s["asset"] != "BTC"]

btc_multi_day, btc_daily = compute_sentiment_scores(btc_signals)
market_multi_day, market_daily = compute_sentiment_scores(market_signals)

# Compute Market Sentiment as the average of all non-BTC assets
market_sentiment = (market_multi_day + market_daily) / 2

# Determine Bias
def get_bias_text(score):
    if score > 0.5:
        return "Bullish"
    elif score < -0.5:
        return "Bearish"
    else:
        return "Mixed"

def get_color(score):
    if score > 0.5:
        return "green"
    elif score < -0.5:
        return "red"
    else:
        return "orange"

btc_color = get_color(btc_multi_day)
market_color = get_color(market_sentiment)

# Determine Bias Text
btc_multi_day_text = get_bias_text(btc_multi_day)
market_sentiment_text = get_bias_text(market_sentiment)

# === 🧠 Sentiment Overview - Positioned Above Technical Signals Summary ===
st.markdown("### 🧠 Sentiment Overview")

col1, col2 = st.columns(2)

# BTC Sentiment Box
col1.markdown(f"""
    <div style='
        background-color: #262730;
        padding: 16px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        text-align: center;
        height: 100%;
    '>
        <div style='font-weight:600; font-size:1.3rem;'>BTC Sentiment</div>
        <div style='font-size:1.8rem; font-weight:700; margin:0.2rem 0; color: {btc_color};'>
            {btc_multi_day_text.upper()}
        </div>
    </div>
""", unsafe_allow_html=True)

# Market Sentiment Box
col2.markdown(f"""
    <div style='
        background-color: #262730;
        padding: 16px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        text-align: center;
        height: 100%;
    '>
        <div style='font-weight:600; font-size:1.3rem;'>Market Sentiment</div>
        <div style='font-size:1.8rem; font-weight:700; margin:0.2rem 0; color: {market_color};'>
            {market_sentiment_text.upper()}
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# === 5️⃣ Technical Signals Summary ===

def interpret_obv(obv_mean):
    """Translate OBV mean to descriptive text."""
    if obv_mean > 0.2:
        return "Strong Accumulation"
    elif 0.1 < obv_mean <= 0.2:
        return "Accumulation"
    elif -0.1 <= obv_mean <= 0.1:
        return "Neutral"
    elif -0.2 <= obv_mean < -0.1:
        return "Distribution"
    else:
        return "Strong Distribution"

def summarize_signals_detailed(signals):
    summary = {
        "Asset": [],
        "Last Cross": [],
        "OBV Mean": [],
        "Last MACD Daily": [],
        "Last MACD Weekly": [],
        "RSI Status": []
    }

    assets = ["BTC", "SP500", "NASDAQ", "GOLD", "DXY"]

    for asset in assets:
        prefix = asset
        asset_df = master_df_dashboard[[col for col in master_df_dashboard.columns if prefix in col]]
        if asset_df.empty:
            continue

        # OBV Mean Interpretation
        obv_col = f"OBV_{prefix}"
        obv_interpretation = "N/A"
        if obv_col in asset_df.columns:
            obv_diff = asset_df[obv_col].diff()
            # Calculate 7-day rolling sum to avoid excessive sensitivity
            obv_rolling = obv_diff.rolling(window=7).sum().dropna()
            last_obv_signal = obv_rolling.iloc[-1]

            if last_obv_signal > 0:
                obv_interpretation = "Accumulation"
            elif last_obv_signal < 0:
                obv_interpretation = "Distribution"

        # Last Cross (Golden/Death) - Track the latest event, regardless of type
        last_cross = "N/A"
        cross_events = []

        for cross_type, col in [("Golden Cross", f"Golden_Cross_Event_{prefix}"), 
                                ("Death Cross", f"Death_Cross_Event_{prefix}")]:
            if col in asset_df.columns:
                event_dates = asset_df[asset_df[col] == 1].index
                if not event_dates.empty:
                    cross_events.append((cross_type, event_dates[-1]))

        if cross_events:
            # Sort by date to get the latest event
            cross_events.sort(key=lambda x: x[1], reverse=True)
            last_cross = f"{cross_events[0][0]} on {cross_events[0][1].date()}"

        # Last MACD Daily & Weekly
        last_macd_daily = "N/A"
        last_macd_weekly = "N/A"

        # Daily MACD
        macd_col_d = f"MACD_D_{prefix}"
        signal_col_d = f"Signal_Line_D_{prefix}"
        if macd_col_d in asset_df.columns and signal_col_d in asset_df.columns:
            macd_series_d = asset_df[macd_col_d]
            signal_series_d = asset_df[signal_col_d]
            
            # Detect crossover events
            bullish_cross_d = (macd_series_d.shift(1) < signal_series_d.shift(1)) & (macd_series_d > signal_series_d)
            bearish_cross_d = (macd_series_d.shift(1) > signal_series_d.shift(1)) & (macd_series_d < signal_series_d)
            
            cross_dates_d = macd_series_d[bullish_cross_d | bearish_cross_d].index.tolist()

            if cross_dates_d:
                last_date_d = cross_dates_d[-1]
                last_macd_value = macd_series_d.loc[last_date_d]
                last_signal_value = signal_series_d.loc[last_date_d]
                macd_type_d = "Bullish" if last_macd_value > last_signal_value else "Bearish"
                last_macd_daily = f"{macd_type_d} Daily MACD Crossover on {last_date_d.date()}"

        # Weekly MACD
        macd_col_w = f"MACD_W_{prefix}"
        signal_col_w = f"Signal_Line_W_{prefix}"
        if macd_col_w in asset_df.columns and signal_col_w in asset_df.columns:
            macd_series_w = asset_df[macd_col_w]
            signal_series_w = asset_df[signal_col_w]

            # Detect crossover events
            bullish_cross_w = (macd_series_w.shift(1) < signal_series_w.shift(1)) & (macd_series_w > signal_series_w)
            bearish_cross_w = (macd_series_w.shift(1) > signal_series_w.shift(1)) & (macd_series_w < signal_series_w)

            cross_dates_w = macd_series_w[bullish_cross_w | bearish_cross_w].index.tolist()

            if cross_dates_w:
                last_date_w = cross_dates_w[-1]
                last_macd_value_w = macd_series_w.loc[last_date_w]
                last_signal_value_w = signal_series_w.loc[last_date_w]
                macd_type_w = "Bullish" if last_macd_value_w > last_signal_value_w else "Bearish"
                last_macd_weekly = f"{macd_type_w} Weekly MACD Crossover on {last_date_w.date()}"

        # RSI Status
        rsi_status = "N/A"
        rsi_overbought_col = f"RSI_Overbought_{prefix}"
        rsi_oversold_col = f"RSI_Oversold_{prefix}"

        if rsi_overbought_col in asset_df.columns and asset_df[rsi_overbought_col].any():
            last_overbought_date = asset_df[asset_df[rsi_overbought_col] == 1].index[-1]
        else:
            last_overbought_date = None

        if rsi_oversold_col in asset_df.columns and asset_df[rsi_oversold_col].any():
            last_oversold_date = asset_df[asset_df[rsi_oversold_col] == 1].index[-1]
        else:
            last_oversold_date = None

        # Determine the most recent RSI event
        if last_overbought_date and last_oversold_date:
            rsi_status = "Overbought" if last_overbought_date > last_oversold_date else "Oversold"
        elif last_overbought_date:
            rsi_status = "Overbought"
        elif last_oversold_date:
            rsi_status = "Oversold"

        # Update summary
        summary["Asset"].append(asset)
        summary["Last Cross"].append(last_cross)
        summary["OBV Mean"].append(obv_interpretation)
        summary["Last MACD Daily"].append(last_macd_daily)
        summary["Last MACD Weekly"].append(last_macd_weekly)
        summary["RSI Status"].append(rsi_status)

    return pd.DataFrame(summary)

# Generate the DataFrame
detailed_summary_df = summarize_signals_detailed(all_signals)
st.dataframe(detailed_summary_df.style.hide(axis="index"))

# === 6️⃣ Signal Legend ===
st.markdown("### 📚 Signal Legend")
signal_explanations = {
    "Golden/Death Cross": "Trend shifts based on moving average crossovers.",
    "Weekly MACD": "Momentum indicator based on weekly MACD.",
    "Daily MACD": "Momentum indicator based on daily MACD.",
    "RSI Signal": "Identifies overbought/oversold conditions.",
    "OBV": "Volume-based trend confirmation."
}
for signal, explanation in signal_explanations.items():
    st.markdown(f"**{signal}:** {explanation}")

# === 7️⃣ Market Sentiment Overview (FNG + Google Trends) ===
st.markdown("### 🌐 Market Sentiment Overview")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 😨 Bitcoin Fear & Greed Index (14D)")
    st.line_chart(master_df_dashboard['Sentiment_BTC_index_value'].tail(14))

with col2:
    st.markdown("### 🔍 Google Trends: 'Bitcoin' (1Y)")
    st.line_chart(google_trends['GT_index_bitcoin'].tail(13))

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
