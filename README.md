# 📉 Swing Scanner

A research screener for oversold rebounds after emotional sell-offs. It finds US stocks that dropped sharply on heavy volume and sit at or below the lower Bollinger Band with a low RSI. For each one it builds a trade plan (entry, stop, target, shares) from your max dollar loss and reward:risk (minimum 1:2) and shows the headlines behind the drop.

Data comes from the unofficial `tradingview-screener` package (about 15 minutes delayed) and Google News RSS. Research only. It never connects to a broker or places orders, and it is not financial advice.

Run locally:

```
pip install -r requirements.txt
streamlit run swing_app.py
```
