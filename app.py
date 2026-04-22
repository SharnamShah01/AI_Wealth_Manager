import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import time
import requests
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError, PermissionDenied, ResourceExhausted
import yfinance as yf

PREFERRED_GEMINI_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite",
    "models/gemini-flash-latest",
    "models/gemini-flash-lite-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]


@st.cache_data(ttl=3600, show_spinner=False)
def list_generate_content_models(api_key: str) -> list[str]:
    genai.configure(api_key=api_key)
    models = []
    for model in genai.list_models():
        methods = getattr(model, "supported_generation_methods", [])
        if "generateContent" in methods:
            models.append(model.name)
    return models


def select_gemini_model(api_key: str) -> str:
    try:
        available = set(list_generate_content_models(api_key))
    except Exception:
        return "models/gemini-2.0-flash"

    for preferred in PREFERRED_GEMINI_MODELS:
        if preferred in available:
            return preferred

    for model_name in sorted(available):
        if "flash" in model_name:
            return model_name

    return sorted(available)[0] if available else "models/gemini-2.0-flash"


def build_model_candidates(api_key: str) -> list[str]:
    try:
        available = set(list_generate_content_models(api_key))
    except Exception:
        available = set()

    candidates = [m for m in PREFERRED_GEMINI_MODELS if m in available]
    if not candidates:
        candidates = ["models/gemini-2.0-flash", "models/gemini-2.0-flash-lite"]

    seen = set()
    ordered = []
    for model_name in candidates:
        if model_name not in seen:
            ordered.append(model_name)
            seen.add(model_name)
    return ordered


def generate_with_retry(api_key: str, prompt: str) -> tuple[str, str]:
    genai.configure(api_key=api_key)
    last_error = None

    for model_name in build_model_candidates(api_key):
        model = genai.GenerativeModel(model_name)
        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                text = getattr(response, "text", None) or "No text was returned by Gemini for this prompt."
                return text, model_name
            except ResourceExhausted as e:
                last_error = e
                time.sleep(2 * (attempt + 1))
            except GoogleAPICallError as e:
                if "429" in str(e):
                    last_error = e
                    time.sleep(2 * (attempt + 1))
                else:
                    raise

    if last_error:
        raise last_error
    raise ResourceExhausted("Rate limit reached across available Gemini models.")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Portfolio Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .metric-card {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .metric-label { font-size: 0.78rem; color: #a0b4c0; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 1.6rem; font-weight: 600; margin: 6px 0; }
    .metric-delta-pos { color: #4ade80; font-size: 0.9rem; }
    .metric-delta-neg { color: #f87171; font-size: 0.9rem; }

    .ai-box {
        background: #0d1117;
        border-left: 4px solid #58a6ff;
        border-radius: 8px;
        padding: 20px 24px;
        margin-top: 12px;
        color: #c9d1d9;
        line-height: 1.75;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 2px solid #e2e8f0;
    }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    div[data-testid="stSidebar"] { background-color: #0f172a; }
    div[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
        width: 100%;
        font-size: 0.9rem;
    }
    .stButton>button:hover { opacity: 0.9; transform: translateY(-1px); transition: all 0.2s; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 24px;'>
    <h1 style='font-size:2rem; font-weight:700; margin:0; color:#0f172a;'>📈 AI Portfolio Advisor</h1>
    <p style='color:#64748b; margin:4px 0 0; font-size:0.95rem;'>
        Live portfolio analytics · Powered by Google Gemini
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar — Portfolio Input
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Portfolio Setup")
    st.markdown("---")

    market = st.radio("Market", ["🇮🇳 NSE (India)", "🇺🇸 US Stocks"], horizontal=True)

    if "NSE" in market:
        st.caption("Use NSE tickers (e.g. RELIANCE, TCS, INFY). `.NS` is added automatically.")
        default_df = pd.DataFrame({
            "Ticker":          ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"],
            "Shares":          [10, 5, 8, 15, 20],
            "Avg Buy Price":   [2800.0, 3500.0, 1600.0, 1700.0, 800.0],
        })
        currency = "₹"
        suffix = ".NS"
    else:
        st.caption("Use standard US tickers (e.g. AAPL, MSFT, GOOGL).")
        default_df = pd.DataFrame({
            "Ticker":          ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
            "Shares":          [10, 8, 5, 6, 7],
            "Avg Buy Price":   [175.0, 380.0, 140.0, 480.0, 340.0],
        })
        currency = "$"
        suffix = ""

    holdings_input = st.data_editor(
        default_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Avg Buy Price": st.column_config.NumberColumn(f"Avg Buy Price ({currency})", format=f"{currency}%.2f"),
            "Shares": st.column_config.NumberColumn("Shares", format="%d"),
        },
        key="holdings_editor",
    )

    st.markdown("---")
    configured_api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or st.secrets.get("GOOGLE_API_KEY", "")
        or st.secrets.get("GEMINI_API_KEY", "")
    )
    api_key = configured_api_key
    analyze_btn = st.button("🤖 Analyze with Gemini AI")

    st.markdown("---")
    st.caption("Google Gemini API")

# ─────────────────────────────────────────────
# Price fetching (cached 5 min)
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(tickers_with_suffix: list[str]) -> dict:
    results = {}
    for t in tickers_with_suffix:
        try:
            hist = yf.Ticker(t).history(period="2d")
            results[t] = float(hist["Close"].iloc[-1]) if not hist.empty else None
        except Exception:
            results[t] = None
    return results


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_exchange_rate(api_key: str, base_currency: str, target_currency: str) -> float:
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{base_currency}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    payload = response.json()

    if payload.get("result") != "success":
        raise ValueError(payload.get("error-type", "Unknown ExchangeRate API error"))

    rates = payload.get("conversion_rates", {})
    rate = rates.get(target_currency)
    if rate is None:
        raise ValueError(f"Exchange rate not available for {base_currency}->{target_currency}")
    return float(rate)


def format_compact_number(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    num = float(value)
    abs_num = abs(num)
    if abs_num >= 1_000_000_000_000:
        return f"{num / 1_000_000_000_000:.2f}T"
    if abs_num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    if abs_num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    if abs_num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return f"{num:.2f}"


def get_quarterly_metrics(ticker: yf.Ticker) -> dict:
    quarterly = ticker.quarterly_income_stmt
    if quarterly is None or quarterly.empty:
        quarterly = ticker.quarterly_financials
    if quarterly is None or quarterly.empty:
        return {}

    def metric_with_growth(metric_name: str):
        if metric_name not in quarterly.index:
            return None, None
        series = quarterly.loc[metric_name].dropna()
        if series.empty:
            return None, None
        latest = float(series.iloc[0])
        previous = float(series.iloc[1]) if len(series) > 1 else None
        growth = ((latest - previous) / abs(previous) * 100) if previous not in (None, 0) else None
        return latest, growth

    revenue, revenue_growth = metric_with_growth("Total Revenue")
    net_income, net_income_growth = metric_with_growth("Net Income")
    return {
        "revenue": revenue,
        "revenue_growth": revenue_growth,
        "net_income": net_income,
        "net_income_growth": net_income_growth,
    }


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_market_context(tickers_with_suffix: list[str]) -> dict:
    context = {}
    for symbol in tickers_with_suffix:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            quarterly = get_quarterly_metrics(ticker)
            headlines = []
            for item in (ticker.news or [])[:3]:
                title = item.get("title")
                if not title:
                    continue
                publisher = item.get("publisher", "Unknown source")
                headlines.append(f"{title} ({publisher})")

            context[symbol] = {
                "company_name": info.get("shortName") or info.get("longName") or symbol,
                "sector": info.get("sector"),
                "market_cap": info.get("marketCap"),
                "trailing_pe": info.get("trailingPE"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "quarterly": quarterly,
                "headlines": headlines,
            }
        except Exception:
            context[symbol] = {}
    return context


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ml_signals(tickers_with_suffix: list[str]) -> dict:
    signals = {}
    for symbol in tickers_with_suffix:
        try:
            hist = yf.Ticker(symbol).history(period="2y", interval="1d")
            if hist is None or hist.empty or "Close" not in hist.columns:
                signals[symbol] = {}
                continue

            close = hist["Close"].dropna()
            returns = close.pct_change()

            feature_df = pd.DataFrame({
                "ret_1d": returns,
                "ret_5d": close.pct_change(5),
                "ret_10d": close.pct_change(10),
                "vol_10d": returns.rolling(10).std(),
                "vol_20d": returns.rolling(20).std(),
                "mom_20d": close / close.rolling(20).mean() - 1,
            })
            target = returns.shift(-1).rename("target_next_day")
            model_df = pd.concat([feature_df, target], axis=1).dropna()

            if len(model_df) < 120:
                signals[symbol] = {}
                continue

            split_idx = int(len(model_df) * 0.8)
            train_df = model_df.iloc[:split_idx]
            test_df = model_df.iloc[split_idx:]
            if test_df.empty:
                signals[symbol] = {}
                continue

            X_train = train_df.drop(columns=["target_next_day"]).to_numpy(dtype=float)
            y_train = train_df["target_next_day"].to_numpy(dtype=float)
            X_test = test_df.drop(columns=["target_next_day"]).to_numpy(dtype=float)
            y_test = test_df["target_next_day"].to_numpy(dtype=float)
            x_latest = model_df.drop(columns=["target_next_day"]).iloc[-1].to_numpy(dtype=float)

            X_train_i = np.column_stack([np.ones(len(X_train)), X_train])
            X_test_i = np.column_stack([np.ones(len(X_test)), X_test])
            x_latest_i = np.concatenate([[1.0], x_latest])

            ridge_lambda = 1e-3
            identity = np.eye(X_train_i.shape[1])
            identity[0, 0] = 0
            weights = np.linalg.solve(
                X_train_i.T @ X_train_i + ridge_lambda * identity,
                X_train_i.T @ y_train
            )

            pred_test = X_test_i @ weights
            pred_next_day = float(x_latest_i @ weights)
            pred_5d = (1 + pred_next_day) ** 5 - 1
            annual_vol = float(returns.dropna().std() * np.sqrt(252))
            directional_accuracy = float(np.mean(np.sign(pred_test) == np.sign(y_test)))
            confidence = float(np.clip((directional_accuracy - 0.45) / 0.25, 0, 1) * 100)

            if annual_vol < 0.2:
                risk_band = "Low"
            elif annual_vol < 0.35:
                risk_band = "Medium"
            else:
                risk_band = "High"

            signals[symbol] = {
                "pred_next_day": pred_next_day,
                "pred_5d": float(pred_5d),
                "annual_vol": annual_vol,
                "directional_accuracy": directional_accuracy,
                "confidence": confidence,
                "risk_band": risk_band,
            }
        except Exception:
            signals[symbol] = {}
    return signals


# ─────────────────────────────────────────────
# Main logic
# ─────────────────────────────────────────────
if holdings_input is None or holdings_input.empty:
    st.info("👈 Enter your portfolio holdings in the sidebar to begin.")
    st.stop()

# Clean & prep
df = holdings_input.dropna(subset=["Ticker"]).copy()
df = df[df["Ticker"].str.strip() != ""]
df["Ticker"] = df["Ticker"].str.upper().str.strip()
df["_lookup"] = df["Ticker"] + suffix

if df.empty:
    st.info("👈 Add at least one valid ticker in the sidebar.")
    st.stop()

# Fetch prices
with st.spinner("📡 Fetching live prices..."):
    raw_prices = fetch_prices(df["_lookup"].tolist())

df["Current Price"] = df["_lookup"].map(raw_prices)

failed = df[df["Current Price"].isna()]["Ticker"].tolist()
if failed:
    st.warning(f"⚠️ Yahoo Finance has no current data for: {', '.join(failed)}. They are excluded from analysis.")

df = df.dropna(subset=["Current Price"])
if df.empty:
    st.error("No valid prices fetched. Please check your tickers.")
    st.stop()

# Derived columns
df["Current Value"] = df["Current Price"] * df["Shares"]
df["Invested"]      = df["Avg Buy Price"] * df["Shares"]
df["PnL"]           = df["Current Value"] - df["Invested"]
df["Return (%)"]    = ((df["Current Price"] - df["Avg Buy Price"]) / df["Avg Buy Price"] * 100).round(2)

total_value    = df["Current Value"].sum()
total_invested = df["Invested"].sum()
total_pnl      = total_value - total_invested
total_return   = total_pnl / total_invested * 100 if total_invested else 0

df["Allocation (%)"] = (df["Current Value"] / total_value * 100).round(2)

with st.spinner("🧠 Running ML forecast model..."):
    ml_signals = fetch_ml_signals(df["_lookup"].drop_duplicates().tolist())

base_currency = "INR" if "NSE" in market else "USD"
target_currency = "USD" if base_currency == "INR" else "INR"
target_symbol = "$" if target_currency == "USD" else "₹"
exchange_rate = None

exchange_rate_api_key = (
    os.getenv("EXCHANGERATE_API_KEY")
    or st.secrets.get("EXCHANGERATE_API_KEY", "")
)

if exchange_rate_api_key:
    try:
        exchange_rate = fetch_exchange_rate(exchange_rate_api_key, base_currency, target_currency)
    except Exception as e:
        st.warning(f"⚠️ Could not fetch {base_currency}->{target_currency} exchange rate: {e}")

# ─────────────────────────────────────────────
# Metric Cards
# ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

def card(col, label, value, delta=None, positive_is_good=True):
    if delta is not None:
        is_pos = delta >= 0
        color = "metric-delta-pos" if (is_pos == positive_is_good) else "metric-delta-neg"
        sign  = "▲" if is_pos else "▼"
        delta_html = f'<div class="{color}">{sign} {abs(delta):.2f}%</div>'
    else:
        delta_html = ""

    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

with c1: card(c1, "Portfolio Value",  f"{currency}{total_value:,.0f}")
with c2: card(c2, "Total Invested",   f"{currency}{total_invested:,.0f}")
with c3: card(c3, "Total P&L",        f"{currency}{total_pnl:,.0f}", total_return)
with c4: card(c4, "Holdings",         str(len(df)))

if exchange_rate:
    converted_value = total_value * exchange_rate
    converted_pnl = total_pnl * exchange_rate
    st.caption(f"FX: 1 {base_currency} = {exchange_rate:.4f} {target_currency}")
    fx1, fx2 = st.columns(2)
    with fx1: card(fx1, f"Portfolio Value ({target_currency})", f"{target_symbol}{converted_value:,.0f}")
    with fx2: card(fx2, f"Total P&L ({target_currency})", f"{target_symbol}{converted_pnl:,.0f}", total_return)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────
ch1, ch2 = st.columns(2)

with ch1:
    fig_pie = px.pie(
        df, values="Current Value", names="Ticker",
        title="Portfolio Allocation",
        hole=0.45,
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig_pie.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value:,.0f}")
    fig_pie.update_layout(showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig_pie, use_container_width=True)

with ch2:
    df_sorted = df.sort_values("Return (%)")
    colors = ["#4ade80" if r >= 0 else "#f87171" for r in df_sorted["Return (%)"]]
    fig_bar = go.Figure(go.Bar(
        x=df_sorted["Ticker"],
        y=df_sorted["Return (%)"],
        marker_color=colors,
        text=[f"{r:.1f}%" for r in df_sorted["Return (%)"]],
        textposition="outside",
    ))
    fig_bar.update_layout(
        title="Individual Returns (%)",
        xaxis_title="Stock", yaxis_title="Return (%)",
        plot_bgcolor="white",
        margin=dict(t=40, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────
# ML Forecast Signals
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">ML Forecast Signals (Experimental)</div>', unsafe_allow_html=True)

ml_rows = []
for _, row in df.drop_duplicates(subset=["_lookup"])[["Ticker", "_lookup"]].iterrows():
    signal = ml_signals.get(row["_lookup"], {})
    if not signal:
        continue
    ml_rows.append({
        "Ticker": row["Ticker"],
        "Predicted Next Day Return": signal["pred_next_day"],
        "Predicted 5D Return": signal["pred_5d"],
        "Annualized Volatility": signal["annual_vol"],
        "Directional Accuracy": signal["directional_accuracy"],
        "Model Confidence": signal["confidence"] / 100.0,
        "Risk Band": signal["risk_band"],
    })

if ml_rows:
    ml_df = pd.DataFrame(ml_rows)
    st.dataframe(
        ml_df.style.format({
            "Predicted Next Day Return": "{:+.2%}",
            "Predicted 5D Return": "{:+.2%}",
            "Annualized Volatility": "{:.2%}",
            "Directional Accuracy": "{:.1%}",
            "Model Confidence": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.caption("Not enough historical data to train ML forecasts for the current tickers.")

# ─────────────────────────────────────────────
# Holdings Table
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Holdings Detail</div>', unsafe_allow_html=True)

display = df[["Ticker", "Shares", "Avg Buy Price", "Current Price",
              "Current Value", "PnL", "Return (%)", "Allocation (%)"]].copy()
display.columns = ["Ticker", "Shares", f"Avg Price ({currency})", f"Current Price ({currency})",
                   f"Value ({currency})", f"P&L ({currency})", "Return (%)", "Allocation (%)"]

def color_pnl(val):
    if isinstance(val, (int, float)):
        return "color: #16a34a; font-weight:600" if val > 0 else ("color: #dc2626; font-weight:600" if val < 0 else "")
    return ""

styled = (
    display.style
    .format({
        f"Avg Price ({currency})":     f"{currency}{{:.2f}}",
        f"Current Price ({currency})": f"{currency}{{:.2f}}",
        f"Value ({currency})":         f"{currency}{{:,.0f}}",
        f"P&L ({currency})":           f"{currency}{{:,.0f}}",
        "Return (%)":                  "{:.2f}%",
        "Allocation (%)":              "{:.2f}%",
    })
    .map(color_pnl, subset=[f"P&L ({currency})", "Return (%)"])
    .set_properties(**{"text-align": "center"})
    .set_table_styles([{
        "selector": "th",
        "props": [("background-color", "#0f172a"), ("color", "white"),
                  ("font-weight", "600"), ("text-align", "center")]
    }])
)
st.dataframe(styled, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# Gemini AI Analysis
# ─────────────────────────────────────────────
if analyze_btn:
    if not api_key:
        st.error("🔑 Google API key is not configured. Add GOOGLE_API_KEY in Streamlit secrets or environment.")
    else:
        holdings_text = df[["Ticker", "Shares", "Avg Buy Price", "Current Price",
                             "Current Value", "PnL", "Return (%)", "Allocation (%)"]].to_string(index=False)
        top_holdings = df.sort_values("Allocation (%)", ascending=False).head(5)[["Ticker", "_lookup", "Allocation (%)"]]

        with st.spinner("📚 Gathering recent news and quarterly data..."):
            market_context = fetch_market_context(top_holdings["_lookup"].tolist())

        context_blocks = []
        for _, row in top_holdings.iterrows():
            ticker_symbol = row["Ticker"]
            lookup_symbol = row["_lookup"]
            allocation = float(row["Allocation (%)"])
            item = market_context.get(lookup_symbol, {})
            company_name = item.get("company_name") or ticker_symbol
            sector = item.get("sector") or "N/A"
            market_cap = format_compact_number(item.get("market_cap"))
            trailing_pe = item.get("trailing_pe")
            trailing_pe_text = f"{float(trailing_pe):.2f}" if trailing_pe is not None else "N/A"

            info_revenue_growth = item.get("revenue_growth")
            info_earnings_growth = item.get("earnings_growth")
            revenue_growth_text = f"{float(info_revenue_growth) * 100:.1f}%" if info_revenue_growth is not None else "N/A"
            earnings_growth_text = f"{float(info_earnings_growth) * 100:.1f}%" if info_earnings_growth is not None else "N/A"

            quarterly = item.get("quarterly", {})
            q_revenue = quarterly.get("revenue")
            q_revenue_growth = quarterly.get("revenue_growth")
            q_income = quarterly.get("net_income")
            q_income_growth = quarterly.get("net_income_growth")

            q_revenue_text = f"{format_compact_number(q_revenue)} ({q_revenue_growth:.1f}% QoQ)" if q_revenue is not None and q_revenue_growth is not None else format_compact_number(q_revenue)
            q_income_text = f"{format_compact_number(q_income)} ({q_income_growth:.1f}% QoQ)" if q_income is not None and q_income_growth is not None else format_compact_number(q_income)
            signal = ml_signals.get(lookup_symbol, {})
            ml_next_day = f"{signal.get('pred_next_day', 0):+.2%}" if signal else "N/A"
            ml_5d = f"{signal.get('pred_5d', 0):+.2%}" if signal else "N/A"
            ml_confidence = f"{signal.get('confidence', 0):.1f}%" if signal else "N/A"
            ml_risk = signal.get("risk_band", "N/A") if signal else "N/A"

            headlines = item.get("headlines") or ["No recent headlines available."]
            headlines_text = "\n".join([f"  - {h}" for h in headlines])

            context_blocks.append(
                f"""{ticker_symbol} ({company_name}) | Allocation: {allocation:.2f}%
- Sector: {sector} | Market Cap: {market_cap} | P/E: {trailing_pe_text}
- Revenue Growth (YoY): {revenue_growth_text} | Earnings Growth (YoY): {earnings_growth_text}
- Latest Quarter Revenue: {q_revenue_text} | Net Income: {q_income_text}
- ML forecast: Next day {ml_next_day}, 5-day {ml_5d}, confidence {ml_confidence}, risk {ml_risk}
- Recent headlines:
{headlines_text}"""
            )

        market_context_text = "\n\n".join(context_blocks)

        prompt = f"""You are a sharp, direct senior financial advisor for high-net-worth individuals.
Analyze the following {'Indian NSE' if 'NSE' in market else 'US'} stock portfolio and give actionable, specific insights.

Portfolio Snapshot:
- Total Value: {currency}{total_value:,.0f}
- Total Invested: {currency}{total_invested:,.0f}
- Overall P&L: {currency}{total_pnl:,.0f} ({total_return:.2f}%)
- Number of Holdings: {len(df)}

Holdings:
{holdings_text}

Additional market context for top holdings (recent headlines + financial fundamentals):
{market_context_text}

Use the ML forecast signals as a quantitative prior, but explicitly mention uncertainty and avoid overconfidence.

Structure your response exactly as follows, using these headings:

### 1. Risk Profile
What type of investor does this portfolio suggest? (aggressive / moderate / conservative). Justify with numbers.

### 2. Concentration Risk
Are any single stocks over-weighted? Any dangerous bets?

### 3. Top Performers & Laggards
What's working? What isn't? Why might that be?

### 4. Rebalancing Recommendations
Give 2–3 specific, actionable suggestions (e.g., trim X, add Y, exit Z). Use allocation percentages.

### 5. Diversification Gaps
What sectors or asset types are missing for a balanced HNI portfolio?

### 6. Overall Verdict
One crisp paragraph: summary + single most important action the investor should take today.

Be direct, use the actual numbers, and write for a sophisticated investor who doesn't need hand-holding."""

        with st.spinner("Gemini is analyzing your portfolio..."):
            try:
                analysis, model_name = generate_with_retry(api_key, prompt)

                st.markdown("---")
                st.markdown('<div class="section-title">🤖 Gemini AI Analysis</div>', unsafe_allow_html=True)
                st.caption(f"Model used: {model_name.replace('models/', '')}")
                st.markdown(f'<div class="ai-box">{analysis}</div>', unsafe_allow_html=True)

            except PermissionDenied:
                st.error("❌ Invalid API key. Please check and try again.")
            except ResourceExhausted:
                st.error("⏳ Rate limit hit on current free quota. Please wait a bit and try again.")
            except GoogleAPICallError as e:
                st.error(f"❌ Google API error: {e}")
            except Exception as e:
                st.error(f"❌ API error: {e}")
