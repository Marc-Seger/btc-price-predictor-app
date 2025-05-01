# 📊 Bitcoin Market Dashboard

An interactive Streamlit dashboard to monitor Bitcoin's price, technical signals, ETF flows, and market sentiment — powered entirely by Python.

## 🚀 Live App & Project Page
👉 [Launch the App](https://marc-seger-bitcoin-market-dashboard.streamlit.app/)
🌐 [Read Full Project Description on My Portfolio](https://marc-seger.github.io/portfolio-website/bitcoin-dashboard.html)

## 🧠 Features

- ✅ Bitcoin price tracking and daily change
- ✅ Fear & Greed Index with 1D and 7D comparisons
- ✅ Latest Spot BTC ETF Net Flows (daily & cumulative)
- ✅ Volume spike detection
- ✅ Dynamic candlestick and line charts (Bitcoin, S&P 500, Nasdaq, DXY, Gold)
- ✅ Technical indicators (SMA, EMA, Bollinger Bands, RSI, MACD, VWAP)
- ✅ Automated signals: Golden/Death Cross, MACD, VWAP trends
- ✅ General market sentiment overview
- ✅ Google Trends & Fear/Greed visualizations

## 📁 Repository Structure
Bitcoin-market-dashboard/
├── data/ # Input data (CSV files from Drive) 
├── docs/ # Workflow and documentation 
├── bitcoin_app.py # Streamlit app code 
├── sync_data_from_drive.py # Script to sync data from Google Drive 
├── requirements.txt # Python dependencies 
├── .streamlit/config.toml # App config (title, theme)

## 🔄 Data Sync

Use `sync_data_from_drive.py` to automatically update CSV data from Google Drive folders.

## 📚 Documentation

See `docs/Bitcoin_Dashboard_Sync_Workflow.txt` for the full sync pipeline and data flow instructions.

## 💡 Next Steps

- Add prediction capabilities (see [btc-price-predictor-app](https://github.com/Marc-Seger/bitcoin-price-predictor-app))
- Expand macro data and trading signal logic
- Build Power BI version of this dashboard

## 🛠️ Tech Stack

- Python · Streamlit · Pandas · Plotly · Google Drive API

---

Created by [Marc Seger](https://marc-seger.github.io/portfolio)
