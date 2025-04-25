import streamlit as st
import pandas as pd
import numpy as np
import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Bitcoin Price Predictor",
    page_icon="📈",
    layout="centered"
)

# --- Title ---
st.title("📈 Bitcoin Price Predictor")
st.markdown("A work-in-progress machine learning app to forecast future Bitcoin prices.")

# --- Sample Line Chart ---
st.subheader("📊 Historical Bitcoin Prices (Sample Data)")

# Generate some sample historical price data
dates = pd.date_range(end=datetime.date.today(), periods=100)
prices = np.cumsum(np.random.randn(100)) * 500 + 30000  # Just a mock-up line chart

btc_df = pd.DataFrame({"Date": dates, "Price": prices})
btc_df = btc_df.set_index("Date")
st.line_chart(btc_df)

# --- Date Input ---
st.subheader("🗓️ Select a Date to Predict")
selected_date = st.date_input(
    "Choose a future date:",
    min_value=datetime.date.today() + datetime.timedelta(days=1),
    max_value=datetime.date.today() + datetime.timedelta(days=60),
    value=datetime.date.today() + datetime.timedelta(days=7)
)

# --- Predict Button (Disabled) ---
st.button("🔮 Predict Price", disabled=True)

st.markdown("> 🚧 *Prediction feature coming soon. Models are being fine-tuned and integrated.*")

# --- Placeholder for future output ---
st.info("The predicted price will appear here when the feature is live.")

# --- Expandable Sections ---
with st.expander("ℹ️ About this project"):
    st.markdown("""
    This app aims to forecast Bitcoin prices using machine learning models like Random Forest and GRU. 
    It's currently under development and will soon support interactive predictions based on real-time financial indicators.
    """)

with st.expander("📦 Data & Models"):
    st.markdown("""
    - Historical data from Yahoo Finance
    - Technical indicators (SMA, EMA, RSI, MACD, etc.)
    - Macroeconomic data (CPI, DXY, Nasdaq, S&P 500)
    - Models: Random Forest (short-term), GRU (long-term)
    """)

with st.expander("🛠️ Roadmap"):
    st.markdown("""
    - [x] Project layout & UI
    - [ ] Integrate trained models
    - [ ] Deploy prediction logic
    - [ ] Continuous model updates
    """)

# --- Footer ---
st.markdown("---")
st.caption("Created by Marc Seger | [GitHub](https://github.com/Marc-Seger)")

