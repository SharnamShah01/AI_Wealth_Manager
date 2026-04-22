# 📈 AI Portfolio Advisor
> Built for Timecell placement drive · Google Gemini API + Streamlit

An AI-powered wealth management tool for analyzing stock portfolios and getting
actionable insights from Google Gemini. Targeted at HNI (High Net-Worth Individual) clients —
directly aligned with Timecell's product vision.

---

## Features

- **Live price fetching** via Yahoo Finance (NSE India + US stocks)
- **Portfolio metrics** — total value, P&L, per-stock returns
- **Live FX conversion** (INR⇄USD) via ExchangeRate API
- **Allocation pie chart** + **returns bar chart** (Plotly)
- **Color-coded holdings table** with full P&L breakdown
- **Gemini AI analysis** — risk profile, concentration risk, rebalancing suggestions,
  diversification gaps, and a crisp overall verdict
- **Context-aware AI insights** using recent stock news + quarterly financial signals
- **Built-in ML forecast model** for next-day / 5-day return and risk band per holding

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. (Optional) Configure Google API key once
Windows PowerShell:
```powershell
$env:GOOGLE_API_KEY="your_google_api_key"
```
Or create `.streamlit/secrets.toml`:
```toml
GOOGLE_API_KEY = "your_google_api_key"
```

### 4. (Optional) Configure ExchangeRate API key once
Add this to `.streamlit/secrets.toml`:
```toml
EXCHANGERATE_API_KEY = "your_exchangerate_api_key"
```

### 5. Usage
- Open http://localhost:8501 in your browser
- Select market (NSE India or US)
- Enter your holdings in the sidebar table (ticker, shares, avg buy price)
- Click **Analyze with Gemini AI** (you can paste key in sidebar, or use configured env key)
- The app auto-selects an available Gemini model for your key/project
- The app auto-fetches INR/USD conversion using ExchangeRate API (if key is configured)

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `streamlit` | UI framework |
| `google-generativeai` | Google Gemini API |
| `yfinance` | Live stock prices |
| `plotly` | Interactive charts |
| `pandas` | Data manipulation |

---

## Example NSE Tickers
`RELIANCE`, `TCS`, `HDFCBANK`, `INFY`, `WIPRO`, `ICICIBANK`, `AXISBANK`, `SBIN`

## Example US Tickers
`AAPL`, `MSFT`, `GOOGL`, `NVDA`, `META`, `AMZN`, `TSLA`, `JPM`

---

## Notes
- API key: Get yours at https://aistudio.google.com/apikey
- Prices are cached for 5 minutes to avoid rate limits
- NSE suffix (`.NS`) is added automatically — just enter the base ticker
