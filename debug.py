from bitcoin_app import all_signals, btc_signals, market_signals, calculate_signal_weight, compute_sentiment_score

# === DEBUGGING SECTION ===

# 1. Print Extracted Signals
print("\n=== Extracted Signals ===")
for signal in all_signals:
    print(signal)

# 2. Check Signal Count by Asset
import pandas as pd
asset_counts = pd.DataFrame(all_signals).groupby("asset").size()
print("\n=== Signal Count by Asset ===")
print(asset_counts)

# 3. Verify BTC and Market Signals
print("\n=== BTC Signals ===")
for signal in btc_signals:
    print(signal)

print("\n=== Market Signals (All Assets) ===")
for signal in market_signals:
    print(signal)

# 4. Check Individual Signal Weights and Contributions
print("\n=== Signal Weights and Contributions ===")
btc_score_details = []
market_score_details = []

for signal in btc_signals:
    weight = signal["weight"]
    date = signal["date"]
    signal_type = signal["type"]
    asset = signal["asset"]
    contribution = weight * calculate_signal_weight(date, signal_type)
    btc_score_details.append({"Asset": asset, "Type": signal_type, "Date": date, "Weight": weight, "Contribution": contribution})

for signal in market_signals:
    weight = signal["weight"]
    date = signal["date"]
    signal_type = signal["type"]
    asset = signal["asset"]
    contribution = weight * calculate_signal_weight(date, signal_type)
    market_score_details.append({"Asset": asset, "Type": signal_type, "Date": date, "Weight": weight, "Contribution": contribution})

# Print Detailed Contributions
print("\n=== BTC Signal Contributions ===")
for entry in btc_score_details:
    print(entry)

print("\n=== Market Signal Contributions ===")
for entry in market_score_details:
    print(entry)

# 5. Display Final Sentiment Scores
btc_sentiment = compute_sentiment_score(btc_signals)
market_sentiment = compute_sentiment_score(market_signals)

print("\n=== Final Sentiment Scores ===")
print(f"BTC Sentiment Score: {btc_sentiment}")
print(f"Market Sentiment Score: {market_sentiment}")
