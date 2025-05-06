import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go   # For dynamic charts
import datetime
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Bitcoin Dashboard",
    page_icon="images/favicon.ico",  
    layout="wide"
)

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

import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# --- Header ---
# === Load base64 image ===
bitcoin_logo_base64 = get_base64_image("images/bitcoin_logo.png")

# === Render title with embedded image ===
# === Refined Logo + Title Alignment ===
last_updated_date = master_df_dashboard.index.max().strftime("%Y-%m-%d")

st.markdown(
    f"""
    <div style='display: flex; align-items: flex-start; gap: 20px; margin-bottom: 25px;'>
        <div style='flex-shrink: 0;'>
            <img src="data:image/png;base64,{bitcoin_logo_base64}" width="70" />
        </div>
        <div>
            <div style='font-size: 3rem; font-weight: 800; color: white; margin-bottom: 6px;'>
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

# --- KPI Cards (Styled) ---
st.subheader("📈 Market Overview")
col1, col2, col3, col4 = st.columns(4)

# === BTC Price ===
btc_series = master_df_dashboard['Close_BTC-USD']
btc_price = btc_series.iloc[-1]

# Find previous different value to avoid flat % change
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

col1.markdown(f"""
    <div style='text-align:center'>
        <div style='font-weight:600; font-size:1.1rem;'>BTC Price</div>
        <div style='font-size:2rem; font-weight:700; margin:0.2rem 0;'>${btc_price:,.0f}</div>
        <div style='font-size:0.9rem; color:{btc_color}; font-weight:500'>{btc_text}</div>
    </div>
""", unsafe_allow_html=True)


# === Fear & Greed Index ===
fng_value = master_df_dashboard['BTC_index_value'].iloc[-1]
fng_label = master_df_dashboard['BTC_index_label'].iloc[-1]

# 1D Change
fng_1d = master_df_dashboard['BTC_index_value'].iloc[-2]
fng_1d_change = fng_value - fng_1d
fng_1d_color = "green" if fng_1d_change > 0 else "red" if fng_1d_change < 0 else "gray"
fng_1d_text = f"{fng_1d_change:+.1f} vs 1D"

# 7D Change
if len(master_df_dashboard) >= 8:
    fng_7d = master_df_dashboard['BTC_index_value'].iloc[-8]
    fng_7d_change = fng_value - fng_7d
    fng_7d_color = "green" if fng_7d_change > 0 else "red" if fng_7d_change < 0 else "gray"
    fng_7d_text = f"{fng_7d_change:+.1f} vs 7D"
else:
    fng_7d_text = "N/A vs 7D"
    fng_7d_color = "gray"

col2.markdown(f"""
    <div style='text-align:center'>
        <div style='font-weight:600; font-size:1.1rem;'>Fear & Greed Index</div>
        <div style='font-size:2rem; font-weight:700; margin:0.2rem 0;'>{fng_value:.1f} ({fng_label})</div>
        <div style='font-size:0.9rem; font-weight:500; display:flex; justify-content:center; gap:12px;'>
            <span style='color:{fng_1d_color}'>{fng_1d_text}</span>
            <span style='color:{fng_7d_color}'>{fng_7d_text}</span>
        </div>
    </div>
""", unsafe_allow_html=True)


# === ETF Net Flow ===
latest_etf_net_flow = etf_flow['Total'].iloc[-1]

col3.markdown(f"""
    <div style='text-align:center'>
        <div style='font-weight:600; font-size:1.1rem;'>Last ETF Net Flow</div>
        <div style='font-size:2rem; font-weight:700; margin:0.2rem 0; color:white'>
            {latest_etf_net_flow:+,.0f}M USD
        </div>
        <div style='font-size:0.9rem; color:gray; font-weight:500'>Latest Daily Value</div>
    </div>
""", unsafe_allow_html=True)


# === Volume Spike ===
volume_spike = master_df_dashboard['High_Volume_BTC'].iloc[-1]
spike_status = "Yes" if volume_spike else "No"
spike_color = "green" if volume_spike else "white"
spike_alert = "🚨" if volume_spike else ""

col4.markdown(f"""
    <div style='text-align:center'>
        <div style='font-weight:600; font-size:1.1rem;'>24h Volume Spike</div>
        <div style='font-size:2rem; font-weight:700; margin:0.2rem 0; color:{spike_color}'>
            {spike_status}
        </div>
        <div style='font-size:0.9rem; color:{spike_color}; font-weight:500'>{spike_alert}</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- Main Chart Section ---
st.subheader("Asset Chart")

# === 1️⃣ Chart Controls ===
asset_options = {
    "Bitcoin": "BTC-USD",
    "S&P 500": "SP500",
    "Nasdaq": "NASDAQ",
    "Gold": "Gold",
    "DXY": "DXY"
}

col1, col2, col3 = st.columns(3)
with col1:
    asset_choice = st.selectbox("Select Asset", list(asset_options.keys()))
with col2:
    timeframe = st.selectbox("Candle Timeframe", ["Daily", "Weekly", "Monthly"])
with col3:
    indicators = st.multiselect("Select Indicators", [
        "SMA_9", "SMA_20", "SMA_50", "SMA_200",
        "EMA_9", "EMA_20", "EMA_50", "EMA_200",
        "Bollinger Bands", "RSI", "MACD"
    ], default=["SMA_50", "SMA_200"])

# Keep chart type on its own row
chart_type = st.radio("Chart Type", ["Line Chart", "Candlestick"], horizontal=True)

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
st.markdown(
    f"<div style='background-color:#14532d; color:white; padding:10px; border-radius:10px;'>"
    f"📊 <strong>Total Cumulative Flow:</strong> {total_flow_billion:,.2f} B USD</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# =========================================
# 🚨 Signals & Insights
# =========================================
st.subheader("🚨 Signals & Insights")

# --- Define Asset Prefixes ---
asset_prefixes = {
    "BTC": "BTC-USD",
    "SP500": "SP500",
    "NASDAQ": "NASDAQ",
    "GOLD": "Gold",
    "DXY": "DXY"
}

def get_asset_data(asset_key):
    prefix = asset_prefixes[asset_key]
    cols = [col for col in master_df_dashboard.columns if f'_{prefix}' in col or col.startswith(f'{prefix}')]
    return master_df_dashboard[cols].copy()

# --- Collect Signal Info ---
summary_data = {"Asset": [], "Signal Summary": [], "Interpretation": []}
detailed_data = []

for asset_key, prefix in asset_prefixes.items():
    df = get_asset_data(asset_key)
    long_term = mid_term = short_term = "Neutral"
    summary_signals = []

    # --- Golden/Death Cross Events ---
    golden_col = f'Golden_Cross_Event_{asset_key}'
    death_col = f'Death_Cross_Event_{asset_key}'

    golden_dates = df[df[golden_col] == 1].index if golden_col in df.columns else []
    death_dates = df[df[death_col] == 1].index if death_col in df.columns else []

    last_golden = golden_dates[-1] if len(golden_dates) > 0 else None
    last_death = death_dates[-1] if len(death_dates) > 0 else None

    if last_golden and (not last_death or last_golden > last_death):
        summary_signals.append("Golden Cross")
        long_term = "Bullish"
        detailed_data.append([asset_key, "Golden Cross", last_golden.strftime('%Y-%m-%d')])
    elif last_death:
        summary_signals.append("Death Cross")
        long_term = "Bearish"
        detailed_data.append([asset_key, "Death Cross", last_death.strftime('%Y-%m-%d')])

    # --- MACD Signal
    macd_signal_col = f'MACD_Above_Signal_{asset_key}'
    if macd_signal_col in df.columns:
        signal_series = df[macd_signal_col]
        latest = signal_series.iloc[-1]
        
        if latest == 1:
            streak_start = signal_series[::-1].ne(1).idxmax()
            summary_signals.append("MACD > Signal Line")
            mid_term = "Bullish"
            detailed_data.append([asset_key, "MACD > Signal Line", streak_start.strftime('%Y-%m-%d')])
        elif latest == 0:
            streak_start = signal_series[::-1].ne(0).idxmax()
            summary_signals.append("MACD < Signal Line")
            mid_term = "Bearish"
            detailed_data.append([asset_key, "MACD < Signal Line", streak_start.strftime('%Y-%m-%d')])

    # --- VWAP Signal
    vwap_signal_col = f'Price_Above_VWAP_{asset_key}'
    if vwap_signal_col in df.columns:
        signal_series = df[vwap_signal_col]
        latest = signal_series.iloc[-1]

        if latest == 1:
            streak_start = signal_series[::-1].ne(1).idxmax()
            summary_signals.append("Price Above VWAP")
            short_term = "Bullish"
            detailed_data.append([asset_key, "Price Above VWAP", streak_start.strftime('%Y-%m-%d')])
        elif latest == 0:
            streak_start = signal_series[::-1].ne(0).idxmax()
            summary_signals.append("Price Below VWAP")
            short_term = "Bearish"
            detailed_data.append([asset_key, "Price Below VWAP", streak_start.strftime('%Y-%m-%d')])

    # --- Interpretation (always executed)
    def map_emoji(val): return "🟢" if val == "Bullish" else "🔴" if val == "Bearish" else "🟠"
    if long_term == short_term == "Bullish":
        interp = f"{map_emoji('Bullish')} Short & Long-Term Bullish"
    elif long_term == short_term == "Bearish":
        interp = f"{map_emoji('Bearish')} Short & Long-Term Bearish"
    elif short_term == "Bullish" and long_term == "Bearish":
        interp = f"🟠 Short-Term Bullish, Long-Term Bearish"
    elif short_term == "Bearish" and long_term == "Bullish":
        interp = f"🟠 Short-Term Bearish, Long-Term Bullish"
    else:
        interp = f"🟠 Mixed Signals – Monitor Closely"

    summary_data["Asset"].append(asset_key)
    summary_data["Signal Summary"].append(", ".join(summary_signals) if summary_signals else "No significant signals")
    summary_data["Interpretation"].append(interp)


# --- Build DataFrames ---
summary_df = pd.DataFrame(summary_data)
detailed_df = pd.DataFrame(detailed_data, columns=["Asset", "Signal Type", "Date"])

# --- Bitcoin Sentiment Box (ETF style)
btc_df = detailed_df[detailed_df["Asset"] == "BTC"]
btc_bull = btc_df["Signal Type"].isin(["Golden Cross", "MACD > Signal Line", "Price Above VWAP"]).sum()
btc_bear = btc_df["Signal Type"].isin(["Death Cross", "MACD < Signal Line", "Price Below VWAP"]).sum()

if btc_bull > btc_bear:
    st.info(f"📢 Bitcoin Sentiment Based on Signals: **Bullish Bias** ({btc_bull} bullish, {btc_bear} bearish signals detected)")
elif btc_bear > btc_bull:
    st.info(f"📢 Bitcoin Sentiment Based on Signals: **Bearish Bias** ({btc_bull} bullish, {btc_bear} bearish signals detected)")
else:
    st.info(f"📢 Bitcoin Sentiment Based on Signals: **Neutral Bias** ({btc_bull} bullish, {btc_bear} bearish signals detected)")

# --- Market Sentiment Box (ETF style)
bullish_assets = detailed_df[detailed_df["Signal Type"].isin(["Golden Cross", "MACD > Signal Line"])]["Asset"].unique()
bearish_assets = detailed_df[detailed_df["Signal Type"].isin(["Death Cross", "MACD < Signal Line"])]["Asset"].unique()

if len(bullish_assets) > len(bearish_assets):
    st.success(f"📢 Market Sentiment Based on Signals: **Bullish Bias** ({len(bullish_assets)} bullish signals: {', '.join(bullish_assets)})")
elif len(bearish_assets) > len(bullish_assets):
    st.error(f"📢 Market Sentiment Based on Signals: **Bearish Bias** ({len(bearish_assets)} bearish signals: {', '.join(bearish_assets)})")
else:
    st.info("📢 Market Sentiment Based on Signals: **Neutral Bias**")

# --- Summary Table ---
st.markdown("### 📊 Technical Signals Summary")
st.dataframe(summary_df, hide_index=True)

# --- Explore Detailed Signals ---
st.markdown("### Explore Detailed Signals")
asset_select = st.selectbox("Select Asset", ["All"] + list(asset_prefixes.keys()))
filtered_df = detailed_df if asset_select == "All" else detailed_df[detailed_df["Asset"] == asset_select]
st.dataframe(filtered_df, hide_index=True)

# --- Signal Legend ---
active_signals = detailed_df["Signal Type"].unique()
signal_explanations = {
    "Golden Cross": "50-day SMA crossing above 200-day SMA → Bullish momentum confirmation.",
    "Death Cross": "50-day SMA crossing below 200-day SMA → Bearish momentum confirmation.",
    "MACD > Signal Line": "MACD line (12,26,9) crossing above Signal Line → Positive trend momentum.",
    "MACD < Signal Line": "MACD line (12,26,9) crossing below Signal Line → Negative trend momentum.",
    "Price Above VWAP": "Price closing above 30-day VWAP → Short-term bullish strength.",
    "Price Below VWAP": "Price closing below 30-day VWAP → Short-term bearish weakness."
}

if active_signals.any():
    st.markdown("### 📚 Signal Legend")
    for signal in active_signals:
        if signal in signal_explanations:
            st.markdown(f"**{signal}:** {signal_explanations[signal]}")

# =========================
# 😨 Sentiment Section
# =========================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 😨 Bitcoin Fear & Greed Index (14D)")
    st.line_chart(master_df_dashboard['BTC_index_value'].tail(14))

with col2:
    st.markdown("### 🔎 Google Trends: 'Bitcoin' (1Y)")
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
