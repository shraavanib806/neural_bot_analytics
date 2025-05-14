# 🧠 Neural Bot Analytics

A real-time stock analytics dashboard that performs **fundamental analysis**, **technical indicator modeling**, and **analyst sentiment breakdowns** — built with **Streamlit**, **yfinance**, and **TA-Lib**.

---

## 📊 Project Overview

This tool allows users to enter any combination of stock tickers (e.g., AAPL, MSFT, GOOGL) and receive:

- 📈 **Fundamental metrics** (market cap, revenue, debt ratios, etc.)
- 📉 **Technical indicators** like RSI, MACD, EMA, momentum
- 🤖 **Technical ratings** engine to identify Buy/Sell signals
- 📊 **Analyst price targets and sentiment**
- 💹 **Visual summaries** of market gainers, losers, undervalued large caps

---

## 🖼️ Screenshots

### Dashboard Overview
![Overview](screenshots/overviews.jpg)

### Earnings Table
![Earnings](screenshots/earnings.jpg)

### Analyst Recommendations
![Analyst Ratings](screenshots/analyst_recs.jpg)

### Market Movers
![Market Movers](./screenshots/market_movers.jpg)

---

## 🛠️ Technologies Used

- Python 3.12
- Streamlit
- yfinance
- pandas
- TA-Lib
- curl_cffi
- lxml-html-clean

---

## ⚙️ How to Run It Locally

```bash
# 1. Clone the repo
git clone https://github.com/shraavanib806/neural_bot_analytics.git
cd neural_bot_analytics

# 2. Create & activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install required libraries
pip install -r requirements.txt
pip install TA-Lib curl_cffi lxml-html-clean

# 4. Run the app
streamlit run neural_bot_beta.py
