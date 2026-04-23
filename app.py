import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from streamlit_local_storage import LocalStorage
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import re
import time
import json
import concurrent.futures
import requests
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError, PermissionDenied, ResourceExhausted
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────
# Browser-local storage — persists across refreshes / sessions
# ─────────────────────────────────────────────────────────────
_ls = LocalStorage()
LS_KEY = "ai_wm_portfolio_v1"

# Each call to _ls.setItem must use a different `key` within the same Streamlit
# run to avoid StreamlitDuplicateElementKey.  We use a simple counter stored in
# session_state so that every invocation gets a unique widget key.
def _save_portfolio(df: pd.DataFrame, mkt: str) -> None:
    """Serialise the portfolio and write it to the browser's LocalStorage."""
    payload = {
        "market": mkt,
        "rows": df.drop(columns=["Remove"], errors="ignore").to_dict(orient="records"),
    }
    # Increment a per-session counter so every setItem call has a unique key.
    _save_ctr = st.session_state.get("_ls_save_ctr", 0) + 1
    st.session_state["_ls_save_ctr"] = _save_ctr
    _ls.setItem(LS_KEY, json.dumps(payload), key=f"_ls_set_{_save_ctr}")

def _load_portfolio() -> tuple:
    """
    Return (df, market) from LocalStorage.
    Returns (None, None) if nothing is saved or data is corrupt.
    """
    try:
        raw = _ls.getItem(LS_KEY)
        if not raw:
            return None, None
        payload = json.loads(raw)
        saved_market = payload.get("market")
        rows = payload.get("rows", [])
        if not rows or not saved_market:
            return None, None
        loaded = pd.DataFrame(rows)
        for col in ["Ticker", "Shares", "Avg Buy Price"]:
            if col not in loaded.columns:
                return None, None
        loaded["Remove"] = False
        loaded["Shares"] = pd.to_numeric(loaded["Shares"], errors="coerce").fillna(0).astype(int)
        loaded["Avg Buy Price"] = pd.to_numeric(loaded["Avg Buy Price"], errors="coerce").fillna(0.0)
        return loaded[["Remove", "Ticker", "Shares", "Avg Buy Price"]], saved_market
    except Exception:
        return None, None

def safe_secret_get(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except (StreamlitSecretNotFoundError, FileNotFoundError):
        return default


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


def stream_with_retry(api_key: str, prompt: str):
    """
    Streams a Gemini response. Returns (chunk_generator, model_name).
    The chunk_generator yields text strings as Gemini produces them.
    """
    genai.configure(api_key=api_key)
    last_error = None

    for model_name in build_model_candidates(api_key):
        model = genai.GenerativeModel(model_name)
        for attempt in range(3):
            try:
                response = model.generate_content(prompt, stream=True)

                def _chunk_gen(resp):
                    for chunk in resp:
                        text = getattr(chunk, "text", None)
                        if text:
                            yield text

                return _chunk_gen(response), model_name
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


def generate_with_retry(
    api_key: str,
    prompt: str,
    generation_config: "genai.types.GenerationConfig | None" = None,
) -> tuple[str, str]:
    """Non-streaming fallback — kept for any future use."""
    genai.configure(api_key=api_key)
    last_error = None

    for model_name in build_model_candidates(api_key):
        model = genai.GenerativeModel(model_name)
        for attempt in range(3):
            try:
                response = model.generate_content(prompt, generation_config=generation_config)
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
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 2px solid rgba(128, 128, 128, 0.3);
    }
    
    /* Hide Streamlit element toolbars (download, search, fullscreen) */
    [data-testid="stElementToolbar"] {
        display: none !important;
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
    <h1 style='font-size:2rem; font-weight:700; margin:0;'>📈 AI Portfolio Advisor</h1>
    <p style='margin:4px 0 0; font-size:0.95rem; opacity:0.8;'>
        Live portfolio analytics · Powered by Google Gemini
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main — Portfolio Input
# ─────────────────────────────────────────────
st.markdown("## ⚙️ Portfolio Setup")
st.markdown("---")

market = st.radio("Market", ["🇮🇳 NSE (India)", "🇺🇸 US Stocks"], horizontal=True)

if "NSE" in market:
    st.caption("Use NSE tickers (e.g. RELIANCE, TCS, INFY). `.NS` is added automatically.")
    default_df = pd.DataFrame({
        "Remove":          [False, False, False, False, False],
        "Ticker":          ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"],
        "Shares":          [10, 5, 8, 15, 20],
        "Avg Buy Price":   [2800.0, 3500.0, 1600.0, 1700.0, 800.0],
    })
    currency = "₹"
    suffix = ".NS"
else:
    st.caption("Use standard US tickers (e.g. AAPL, MSFT, GOOGL).")
    default_df = pd.DataFrame({
        "Remove":          [False, False, False, False, False],
        "Ticker":          ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
        "Shares":          [10, 8, 5, 6, 7],
        "Avg Buy Price":   [175.0, 380.0, 140.0, 480.0, 340.0],
    })
    currency = "$"
    suffix = ""

# ── State initialization & LocalStorage Sync ────────────────────────────────
if "portfolio_df" not in st.session_state:
    # First-ever run in this session: start with defaults
    st.session_state.portfolio_df = default_df.copy()
    st.session_state.market = market
    st.session_state.deleted_rows = []
    st.session_state.ls_synced = False

# Try to sync from browser LocalStorage if not already done
if not st.session_state.ls_synced:
    ls_df, ls_market = _load_portfolio()
    if ls_df is not None:
        # Success! Browser has data. Overwrite defaults.
        st.session_state.portfolio_df = ls_df
        st.session_state.market = ls_market
        st.session_state.ls_synced = True
        st.toast("✅ Portfolio restored from browser storage!", icon="💾")
        st.rerun()

# If the user manually toggles the Market radio button, reset to that market's defaults.
# We do NOT call _save_portfolio here (that would collide with the auto-save below in
# the same script run and trigger StreamlitDuplicateElementKey).  Instead we set a flag
# and let the auto-save block handle it after rerun.
if st.session_state.get("market") != market:
    st.session_state.portfolio_df = default_df.copy()
    st.session_state.market = market
    st.session_state.deleted_rows = []
    st.session_state.ls_synced = True
    st.session_state.last_saved_state = ""  # force auto-save to trigger on next run
    st.rerun()

# ── Undo Button ─────────────────────────────────────────────────────────────
if st.session_state.get("deleted_rows"):
    if st.button(f"↩️ Undo Delete ({len(st.session_state.deleted_rows)} available)"):
        last_deleted = st.session_state.deleted_rows.pop()
        st.session_state.portfolio_df = pd.concat(
            [st.session_state.portfolio_df, last_deleted], ignore_index=True
        )
        _save_portfolio(st.session_state.portfolio_df, market)
        st.rerun()

# ── Data Editor ──────────────────────────────────────────────────────────────
editor_key = f"holdings_editor_{len(st.session_state.portfolio_df)}"

edited_df = st.data_editor(
    st.session_state.portfolio_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Remove": st.column_config.CheckboxColumn("🗑️", default=False),
        "Avg Buy Price": st.column_config.NumberColumn(f"Avg Buy Price ({currency})", format=f"{currency}%.2f"),
        "Shares": st.column_config.NumberColumn("Shares", format="%d"),
    },
    key=editor_key,
)

# ── Process deletions ────────────────────────────────────────────────────────
removed_mask = edited_df["Remove"] == True
if removed_mask.any():
    removed_rows = edited_df[removed_mask].copy()
    removed_rows["Remove"] = False
    st.session_state.deleted_rows.append(removed_rows)
    kept_rows = edited_df[~removed_mask].copy()
    st.session_state.portfolio_df = kept_rows
    _save_portfolio(kept_rows, market)
    st.rerun()

holdings_input = edited_df.drop(columns=["Remove"])

# ── Silent Auto-Save ─────────────────────────────────────────────────────────
# Compare the current editor state to the last saved state to trigger autosaves
# without mutating st.session_state.portfolio_df (which breaks UI focus)
current_state_str = edited_df.to_json()
if "last_saved_state" not in st.session_state:
    st.session_state.last_saved_state = current_state_str

if current_state_str != st.session_state.last_saved_state:
    _save_portfolio(edited_df, market)
    st.session_state.last_saved_state = current_state_str
    st.session_state.ls_synced = True

st.caption("💡 **Tip:** Check the **🗑️ box** to instantly remove a stock, or click the empty bottom row to add a new one.")
st.caption("🟢 *Auto-saving in real-time to your browser*")

st.markdown("---")

configured_api_key = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or safe_secret_get("GOOGLE_API_KEY", "")
    or safe_secret_get("GEMINI_API_KEY", "")
)

api_key = configured_api_key
analyze_btn = st.button("🤖 Analyze with Gemini AI")

st.markdown("---")


# ─────────────────────────────────────────────
# Price fetching (cached 5 min)
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(tickers_with_suffix: list[str]) -> dict:
    results = {}
    for t in tickers_with_suffix:
        try:
            hist = yf.Ticker(t).history(period="7d")
            valid_closes = hist["Close"].dropna()
            results[t] = float(valid_closes.iloc[-1]) if not valid_closes.empty else None
        except Exception:
            results[t] = None
    return results

def resolve_ticker(query: str) -> str:
    """Uses Yahoo Finance Search API to find the best official ticker for a given text."""
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={requests.utils.quote(query)}&quotesCount=1"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        res.raise_for_status()
        quotes = res.json().get("quotes", [])
        if quotes:
            symbol = quotes[0].get("symbol")
            if symbol:
                return symbol
    except Exception:
        pass
    return query


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


def classify_market_cap(mc: float, is_nse: bool) -> str:
    if not mc or np.isnan(mc): return "Unknown"
    # INR Classification (in Crores: 1 Cr = 10_000_000)
    if is_nse:
        if mc > 200_000_000_000: return "Large Cap"
        if mc > 50_000_000_000: return "Mid Cap"
        if mc > 5_000_000_000: return "Small Cap"
        return "Micro Cap"
    else:
        # USD Classification
        if mc > 10_000_000_000: return "Large Cap"
        if mc > 2_000_000_000: return "Mid Cap"
        if mc > 300_000_000: return "Small Cap"
        return "Micro Cap"


def get_quarterly_metrics(ticker: yf.Ticker) -> dict:
    quarterly_is = ticker.quarterly_income_stmt
    if quarterly_is is None or quarterly_is.empty:
        quarterly_is = ticker.quarterly_financials
        
    quarterly_bs = ticker.quarterly_balance_sheet

    def metric_with_growth(df, metric_name: str):
        if df is None or df.empty or metric_name not in df.index:
            return None, None
        series = df.loc[metric_name].dropna()
        if series.empty:
            return None, None
        latest = float(series.iloc[0])
        previous = float(series.iloc[1]) if len(series) > 1 else None
        growth = ((latest - previous) / abs(previous) * 100) if previous not in (None, 0) else None
        return latest, growth

    revenue, revenue_growth = metric_with_growth(quarterly_is, "Total Revenue")
    net_income, net_income_growth = metric_with_growth(quarterly_is, "Net Income")
    
    total_debt, _ = metric_with_growth(quarterly_bs, "Total Debt")
    total_equity, _ = metric_with_growth(quarterly_bs, "Stockholders Equity")
    
    debt_to_equity = (total_debt / total_equity) if total_debt is not None and total_equity not in (None, 0) else None
    profit_margin = (net_income / revenue * 100) if net_income is not None and revenue not in (None, 0) else None
    roe = (net_income / total_equity * 100) if net_income is not None and total_equity not in (None, 0) else None

    return {
        "revenue": revenue,
        "revenue_growth": revenue_growth,
        "net_income": net_income,
        "net_income_growth": net_income_growth,
        "debt_to_equity": debt_to_equity,
        "profit_margin": profit_margin,
        "roe": roe,
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

            mc = info.get("marketCap")
            cap_tier = classify_market_cap(mc, symbol.endswith(".NS"))

            context[symbol] = {
                "company_name": info.get("shortName") or info.get("longName") or symbol,
                "sector": info.get("sector"),
                "market_cap": mc,
                "market_cap_tier": cap_tier,
                "trailing_pe": info.get("trailingPE"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "quarterly": quarterly,
                "headlines": headlines,
            }
        except Exception:
            context[symbol] = {}
    return context


# ─────────────────────────────────────────────
# Confidence Rating System
# ─────────────────────────────────────────────
def calculate_data_quality_score(df: pd.DataFrame, ml_signals: dict) -> float:
    """
    Scores data quality based on:
    - Percentage of holdings with valid prices
    - Percentage of holdings with ML signals
    Returns: 0-100 score
    """
    if df.empty:
        return 0.0
    
    price_coverage = (df["Current Price"].notna().sum() / len(df)) * 100
    
    ml_signal_count = 0
    for ticker in df["_lookup"].unique():
        if ticker in ml_signals and ml_signals[ticker]:
            ml_signal_count += 1
    ml_coverage = (ml_signal_count / len(df.drop_duplicates(subset=["_lookup"]))) * 100
    
    data_quality = (price_coverage * 0.6 + ml_coverage * 0.4)
    return float(np.clip(data_quality, 0, 100))


def calculate_market_stability_score(df: pd.DataFrame, ml_signals: dict) -> float:
    """
    Scores market stability based on:
    - Portfolio volatility (lower is better)
    - Consistency of ML directional accuracy
    Returns: 0-100 score (higher = more stable, more predictable)
    """
    if df.empty or not ml_signals:
        return 50.0
    
    volatilities = []
    accuracies = []
    
    for ticker in df["_lookup"].unique():
        signal = ml_signals.get(ticker, {})
        if signal:
            if "annual_vol" in signal:
                volatilities.append(signal["annual_vol"])
            if "directional_accuracy" in signal:
                accuracies.append(signal["directional_accuracy"])
    
    if not volatilities:
        return 50.0
    
    avg_vol = float(np.mean(volatilities))
    avg_accuracy = float(np.mean(accuracies)) if accuracies else 0.5
    
    vol_score = max(0, 100 * (1 - avg_vol / 1.0))
    accuracy_score = min(100, 50 + avg_accuracy * 100)
    
    market_stability = (vol_score * 0.5 + accuracy_score * 0.5)
    return float(np.clip(market_stability, 0, 100))


def calculate_diversification_score(df: pd.DataFrame) -> float:
    """
    Calculates portfolio diversification using Herfindahl index.
    Returns: 0-100 score (100 = perfectly diversified, 0 = concentrated)
    """
    if df.empty or df["Allocation (%)"].sum() == 0:
        return 0.0
    
    allocations = df["Allocation (%)"].values / 100.0
    hhi = float(np.sum(allocations ** 2))
    
    max_holdings = len(df)
    min_hhi = 1.0 / max_holdings if max_holdings > 0 else 1.0
    max_hhi = 1.0
    
    normalized_hhi = (max_hhi - hhi) / (max_hhi - min_hhi) if max_hhi > min_hhi else 0.0
    diversification_score = max(0, min(100, normalized_hhi * 100))
    
    return float(diversification_score)


def calculate_ml_strength_score(ml_signals: dict) -> float:
    """
    Scores ML model strength based on:
    - Number of valid signals
    - Average directional accuracy
    - Consistency of predictions
    Returns: 0-100 score
    """
    if not ml_signals:
        return 30.0
    
    valid_signals = [s for s in ml_signals.values() if s and "directional_accuracy" in s]
    
    if not valid_signals:
        return 30.0
    
    signal_count_score = min(100, len(valid_signals) * 15)
    
    accuracies = [s["directional_accuracy"] for s in valid_signals]
    avg_accuracy = float(np.mean(accuracies))
    accuracy_score = (avg_accuracy - 0.5) * 200
    accuracy_score = float(np.clip(accuracy_score, 0, 100))
    
    confidence_values = [s.get("confidence", 50) / 100.0 for s in valid_signals]
    avg_confidence = float(np.mean(confidence_values))
    confidence_score = avg_confidence * 100
    
    ml_strength = (signal_count_score * 0.3 + accuracy_score * 0.4 + confidence_score * 0.3)
    return float(np.clip(ml_strength, 0, 100))


def calculate_overall_confidence(
    df: pd.DataFrame,
    ml_signals: dict,
    total_value: float,
    total_invested: float
) -> dict:
    """
    Calculates overall analysis confidence rating.
    Returns: dict with confidence score (0-100) and component breakdown
    """
    if df.empty or total_value == 0:
        return {
            "score": 0.0,
            "data_quality": 0.0,
            "market_stability": 50.0,
            "diversification": 0.0,
            "ml_strength": 30.0,
            "interpretation": "Insufficient data"
        }
    
    data_quality = calculate_data_quality_score(df, ml_signals)
    market_stability = calculate_market_stability_score(df, ml_signals)
    diversification = calculate_diversification_score(df)
    ml_strength = calculate_ml_strength_score(ml_signals)
    
    overall_confidence = (
        data_quality * 0.25 +
        market_stability * 0.25 +
        diversification * 0.25 +
        ml_strength * 0.25
    )
    
    if overall_confidence >= 75:
        interpretation = "High confidence — data is comprehensive and portfolio is well-positioned"
    elif overall_confidence >= 55:
        interpretation = "Moderate confidence — sufficient data but some limitations exist"
    else:
        interpretation = "Low confidence — limited data or high volatility; recommendations are preliminary"
    
    return {
        "score": float(np.clip(overall_confidence, 0, 100)),
        "data_quality": float(data_quality),
        "market_stability": float(market_stability),
        "diversification": float(diversification),
        "ml_strength": float(ml_strength),
        "interpretation": interpretation
    }


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ml_signals(tickers_with_suffix: list[str]) -> dict:
    """
    Computes ML-based directional forecasts for each ticker using a
    RandomForestClassifier trained on rich technical indicators:
      RSI-14, MACD, MACD histogram, Bollinger Band position,
      EMA20/EMA50 crossover, ATR-14, OBV, volume ratio,
      1d/5d/10d/20d returns, 10d/20d volatility.
    Confidence = predict_proba (probability of the predicted direction).
    """
    def _ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    def _rsi(close, period=14):
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    def _atr(high, low, close, period=14):
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _bollinger_pos(close, period=20, n_std=2):
        """Position within Bollinger Bands: 0 = lower band, 1 = upper band."""
        mid   = close.rolling(period).mean()
        std   = close.rolling(period).std()
        upper = mid + n_std * std
        lower = mid - n_std * std
        width = (upper - lower).replace(0, np.nan)
        return (close - lower) / width

    signals = {}
    for symbol in tickers_with_suffix:
        try:
            ticker_obj = yf.Ticker(symbol)
            hist = ticker_obj.history(period="2y", interval="1d")

            # ── Detect unlisted / no-data vs. thin-data ──
            if hist is None or hist.empty or "Close" not in hist.columns:
                signals[symbol] = {"status": "no_data"}   # private / unlisted
                continue

            close  = hist["Close"].dropna()
            volume = hist["Volume"] if "Volume" in hist.columns else pd.Series(dtype=float)
            high   = hist["High"]   if "High"   in hist.columns else close
            low    = hist["Low"]    if "Low"    in hist.columns else close

            if len(close) < 100:
                signals[symbol] = {"status": "thin_data", "days": len(close)}
                continue

            returns = close.pct_change()

            # ── Feature engineering ──
            rsi       = _rsi(close)
            ema12     = _ema(close, 12)
            ema26     = _ema(close, 26)
            macd      = ema12 - ema26
            macd_sig  = _ema(macd, 9)
            macd_hist = macd - macd_sig
            ema20     = _ema(close, 20)
            ema50     = _ema(close, 50)
            ema_cross = ema20 / ema50.replace(0, np.nan) - 1   # >0 = golden cross
            bb_pos    = _bollinger_pos(close)
            atr       = _atr(high, low, close)
            atr_pct   = atr / close                             # normalised ATR

            if len(volume.dropna()) > 10:
                vol_ma    = volume.rolling(20).mean()
                vol_ratio = volume / vol_ma.replace(0, np.nan)
            else:
                vol_ratio = pd.Series(1.0, index=close.index)

            # OBV (On-Balance Volume) momentum
            obv_delta = np.sign(returns) * volume
            obv_mom   = obv_delta.rolling(10).sum()

            feature_df = pd.DataFrame({
                "ret_1d":    returns,
                "ret_5d":    close.pct_change(5),
                "ret_10d":   close.pct_change(10),
                "ret_20d":   close.pct_change(20),
                "vol_10d":   returns.rolling(10).std(),
                "vol_20d":   returns.rolling(20).std(),
                "rsi":       rsi,
                "macd":      macd,
                "macd_hist": macd_hist,
                "ema_cross": ema_cross,
                "bb_pos":    bb_pos,
                "atr_pct":   atr_pct,
                "vol_ratio": vol_ratio,
                "obv_mom":   obv_mom,
            }, index=close.index)

            # Target: 1 if next-day return > 0, else 0 (classification)
            target = (returns.shift(-1) > 0).astype(int).rename("target")
            model_df = pd.concat([feature_df, target], axis=1).dropna()

            if len(model_df) < 100:
                signals[symbol] = {"status": "thin_data", "days": len(model_df)}
                continue

            X = model_df.drop(columns=["target"]).to_numpy(dtype=float)
            y = model_df["target"].to_numpy(dtype=int)

            # ── Time-series cross-validation ──
            tscv    = TimeSeriesSplit(n_splits=5)
            da_list = []   # directional accuracy per fold
            pb_list = []   # probability of predicted class per fold

            scaler = StandardScaler()
            rf     = RandomForestClassifier(
                n_estimators=120,
                max_depth=6,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )

            for train_idx, test_idx in tscv.split(X):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]
                if len(np.unique(y_tr)) < 2:
                    continue   # skip degenerate fold
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)
                rf.fit(X_tr_s, y_tr)
                preds  = rf.predict(X_te_s)
                probas = rf.predict_proba(X_te_s)
                da_list.append(float(np.mean(preds == y_te)))
                # confidence = average probability of the predicted class
                pb_list.append(float(np.mean(probas[np.arange(len(preds)), preds])))

            if not da_list:
                signals[symbol] = {"status": "thin_data", "days": len(model_df)}
                continue

            directional_accuracy = float(np.mean(da_list))
            avg_prob             = float(np.mean(pb_list))
            # Confidence: how far above 50% (random) the avg probability is,
            # scaled so 50% prob → 0% confidence, 75% prob → 100% confidence
            confidence = float(np.clip((avg_prob - 0.50) / 0.25, 0, 1) * 100)

            # ── Final prediction on latest data ──
            X_all_s   = scaler.fit_transform(X)
            rf.fit(X_all_s, y)
            x_latest  = X[-1].reshape(1, -1)
            x_latest_s = scaler.transform(x_latest)
            pred_dir   = int(rf.predict(x_latest_s)[0])
            pred_prob  = float(rf.predict_proba(x_latest_s)[0][pred_dir])

            # Sign-and-magnitude: use recent volatility to estimate magnitude
            annual_vol    = float(returns.dropna().std() * np.sqrt(252))
            daily_vol     = annual_vol / np.sqrt(252)
            pred_next_day = (1 if pred_dir == 1 else -1) * daily_vol * pred_prob
            pred_5d       = (1 + pred_next_day) ** 5 - 1

            if annual_vol < 0.20:
                risk_band = "Low"
            elif annual_vol < 0.35:
                risk_band = "Medium"
            else:
                risk_band = "High"

            # Top-3 feature importances (human-readable)
            feat_names   = list(feature_df.columns)
            importances  = rf.feature_importances_
            top_feats    = sorted(zip(feat_names, importances), key=lambda x: -x[1])[:3]
            top_feat_str = ", ".join(f"{n} ({v:.0%})" for n, v in top_feats)

            signals[symbol] = {
                "status":               "ok",
                "pred_next_day":        pred_next_day,
                "pred_5d":              float(pred_5d),
                "annual_vol":           annual_vol,
                "directional_accuracy": directional_accuracy,
                "confidence":           confidence,
                "risk_band":            risk_band,
                "top_features":         top_feat_str,
                "avg_prob":             avg_prob,
            }
        except Exception:
            signals[symbol] = {"status": "error"}
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
    
    # Smart Ticker Fallback
    for idx, row in df.iterrows():
        t = row["_lookup"]
        if raw_prices.get(t) is None:
            new_symbol = resolve_ticker(row["Ticker"])
            if new_symbol and new_symbol != t and new_symbol != row["Ticker"]:
                fb_price = fetch_prices([new_symbol])
                if fb_price.get(new_symbol) is not None:
                    df.at[idx, "_lookup"] = new_symbol
                    df.at[idx, "Ticker"] = new_symbol.replace(suffix, "") if suffix else new_symbol
                    raw_prices[new_symbol] = fb_price[new_symbol]
                    st.toast(f"Auto-resolved '{row['Ticker']}' to '{new_symbol}'", icon="🔍")

df["Current Price"] = df["_lookup"].map(raw_prices)

failed = df[df["Current Price"].isna()]["Ticker"].tolist()
if failed:
    # Hint that some might be private/unlisted rather than just a bad ticker
    private_hint = " Some of these may be private/unlisted companies (e.g. Zepto, CRED) that have no stock exchange listing - those cannot be analysed with market data." if len(failed) > 0 else ""
    st.warning(
        f"⚠️ Could not fetch market data for: {', '.join(failed)}. "
        f"They have been excluded from the analysis."
        f"{private_hint}"
    )

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

with st.spinner("📊 Computing analysis confidence..."):
    confidence_data = calculate_overall_confidence(df, ml_signals, total_value, total_invested)

base_currency = "INR" if "NSE" in market else "USD"
target_currency = "USD" if base_currency == "INR" else "INR"
target_symbol = "$" if target_currency == "USD" else "₹"
exchange_rate = None

exchange_rate_api_key = (
    os.getenv("EXCHANGERATE_API_KEY")
    or safe_secret_get("EXCHANGERATE_API_KEY", "")
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
# AI Confidence Rating shown after Gemini responds
# ─────────────────────────────────────────────
# (Gemini self-assesses its confidence as part of its JSON response — see Gemini section below)

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
    fig_pie.update_layout(
        showlegend=False, 
        margin=dict(t=40, b=10, l=10, r=10),
        dragmode=False  # Disables dragging/zooming
    )
    st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

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
        xaxis_title="Stock", 
        yaxis_title="Return (%)",
        plot_bgcolor="white",
        margin=dict(t=40, b=10, l=10, r=10),
        dragmode=False,  # Disables box zoom
        xaxis=dict(fixedrange=True),  # Disables pinch/scroll zoom
        yaxis=dict(fixedrange=True)
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})


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
    st.toast("⏳ Analysis started — scroll down to see results!", icon="🤖")
    if not api_key:
        st.error("🔑 Google API key is not configured. Add GOOGLE_API_KEY in Streamlit secrets or environment.")
    else:
        # Persist any manually typed edits to LocalStorage now
        _save_portfolio(edited_df, market)
        st.session_state.ls_synced = True

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
            cap_tier = item.get("market_cap_tier", "Unknown")
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
            q_debt_to_equity = quarterly.get("debt_to_equity")
            q_profit_margin = quarterly.get("profit_margin")
            q_roe = quarterly.get("roe")

            q_revenue_text = f"{format_compact_number(q_revenue)} ({q_revenue_growth:.1f}% QoQ)" if q_revenue is not None and q_revenue_growth is not None else format_compact_number(q_revenue)
            q_income_text = f"{format_compact_number(q_income)} ({q_income_growth:.1f}% QoQ)" if q_income is not None and q_income_growth is not None else format_compact_number(q_income)
            
            q_de_text = f"{q_debt_to_equity:.2f}" if q_debt_to_equity is not None else "N/A"
            q_pm_text = f"{q_profit_margin:.1f}%" if q_profit_margin is not None else "N/A"
            q_roe_text = f"{q_roe:.1f}%" if q_roe is not None else "N/A"

            signal      = ml_signals.get(lookup_symbol, {})
            ok          = signal.get("status") == "ok"
            ml_next_day = f"{signal.get('pred_next_day', 0):+.2%}" if ok else "N/A"
            ml_5d       = f"{signal.get('pred_5d', 0):+.2%}"       if ok else "N/A"
            ml_confidence = f"{signal.get('confidence', 0):.0f}% (prob {signal.get('avg_prob', 0):.0%})" if ok else "N/A"
            ml_risk     = signal.get("risk_band", "N/A")            if ok else "N/A"
            ml_drivers  = signal.get("top_features", "N/A")         if ok else "N/A"

            headlines = item.get("headlines") or ["No recent headlines available."]
            headlines_text = "\n".join([f"  - {h}" for h in headlines])

            context_blocks.append(
                f"""{ticker_symbol} ({company_name}) | Allocation: {allocation:.2f}%
- Sector: {sector} | Market Cap: {market_cap} ({cap_tier}) | P/E: {trailing_pe_text}
- Revenue Growth (YoY): {revenue_growth_text} | Earnings Growth (YoY): {earnings_growth_text}
- Latest Quarter Revenue: {q_revenue_text} | Net Income: {q_income_text}
- Profit Margin: {q_pm_text} | ROE: {q_roe_text} | Debt-to-Equity: {q_de_text}
- ML forecast (Random Forest, 14 indicators): Next day {ml_next_day}, 5-day {ml_5d}, confidence {ml_confidence}, risk {ml_risk}
- ML key drivers: {ml_drivers}
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

Use the ML forecast signals as a quantitative prior, but mention uncertainty where appropriate.

Structure your entire response in clean, readable Markdown using EXACTLY these sections in this order:

## 1. Risk Profile
What type of investor does this portfolio suggest? (aggressive / moderate / conservative). Justify with numbers.

## 2. Concentration Risk
Are any single stocks over-weighted? Any dangerous bets?

## 3. Top Performers & Laggards
What's working? What isn't? Why might that be?

## 4. Rebalancing Recommendations
Give 2–3 specific, actionable suggestions (e.g., trim X, add Y, exit Z). Use allocation percentages.

## 5. Diversification Gaps
What sectors or asset types are missing for a balanced HNI portfolio?

## 6. Overall Verdict
One crisp paragraph: summary + single most important action the investor should take today.

After section 6, on its own line, output EXACTLY this format (do not omit it):
CONFIDENCE_SCORE: <integer 0-100>
CONFIDENCE_REASON: <one or two sentences on why you are that confident>

Be direct, use the actual numbers, and write for a sophisticated investor who doesn't need hand-holding."""


        # ── Build a dynamic, factually-accurate step list ──────────────────────
        # Every step references real data that was actually computed above.
        ANALYSIS_STEPS = []

        # Phase 1 – Setup
        ANALYSIS_STEPS.append(("🔗", "Connecting to Gemini AI..."))
        ANALYSIS_STEPS.append(("📂", f"Loading portfolio: {len(df)} holding{'s' if len(df) != 1 else ''} "
                                      f"— total value {currency}{total_value:,.0f}"))

        # Phase 2 – Per-stock portfolio review (one step per ticker)
        for _, row in df.iterrows():
            ret_str = f"{row['Return (%)']:+.1f}%"
            alloc_str = f"{row['Allocation (%)']:.1f}%"
            ANALYSIS_STEPS.append((
                "📈",
                f"Reviewing {row['Ticker']}: price {currency}{row['Current Price']:.2f} "
                f"| return {ret_str} | allocation {alloc_str}"
            ))

        # Phase 3 – ML signal review (one step per ticker)
        for _, row in df.drop_duplicates(subset=["_lookup"]).iterrows():
            signal = ml_signals.get(row["_lookup"], {})
            if signal:
                risk  = signal.get("risk_band", "N/A")
                dacc  = signal.get("directional_accuracy", 0)
                conf  = signal.get("confidence", 0)
                vol   = signal.get("annual_vol", 0)
                # Show a plain-English confidence label instead of a misleading 0%
                if conf >= 60:
                    conf_text = f"{conf:.0f}% model confidence"
                elif conf >= 20:
                    conf_text = f"low confidence ({conf:.0f}%)"
                else:
                    conf_text = "no directional edge (model at or below random chance)"
                ANALYSIS_STEPS.append((
                    "🤖",
                    f"ML signals for {row['Ticker']}: {risk} risk | "
                    f"{dacc:.0%} directional accuracy | {conf_text} | "
                    f"{vol:.0%} annual volatility"
                ))
            else:
                ANALYSIS_STEPS.append((
                    "🤖",
                    f"ML signals for {row['Ticker']}: no price data available — "
                    "this ticker may be unlisted or private; Gemini will use fundamentals only"
                ))

        # Phase 4 – News review (one step per top holding)
        for _, row in top_holdings.iterrows():
            item      = market_context.get(row["_lookup"], {})
            company   = item.get("company_name") or row["Ticker"]
            headlines = item.get("headlines") or []
            n         = len(headlines)
            if n > 0:
                ANALYSIS_STEPS.append((
                    "📰",
                    f"Reviewed {n} recent headline{'s' if n != 1 else ''} for {company} "
                    f"({row['Ticker']}) — {row['Allocation (%)']:.1f}% of portfolio"
                ))
            else:
                ANALYSIS_STEPS.append((
                    "📰",
                    f"Checking market news & sentiment for {company} ({row['Ticker']}) "
                    f"— {row['Allocation (%)']:.1f}% of portfolio"
                ))

        # Phase 5 – Fundamentals review (one step per top holding)
        for _, row in top_holdings.iterrows():
            item    = market_context.get(row["_lookup"], {})
            company = item.get("company_name") or row["Ticker"]
            sector  = item.get("sector") or "N/A"
            pe      = item.get("trailing_pe")
            pe_text = f"P/E {float(pe):.1f}" if pe is not None else "P/E unavailable"
            mcap    = format_compact_number(item.get("market_cap"))
            ANALYSIS_STEPS.append((
                "📡",
                f"Fundamentals for {company}: {sector} sector | {pe_text} | Mkt cap {mcap}"
            ))

        # Phase 6 – Cross-referencing ML vs. news sentiment
        ANALYSIS_STEPS.append(("🔄", "Cross-referencing ML forecasts against recent headline sentiment..."))
        ANALYSIS_STEPS.append(("📊", "Calculating effective sector exposure across all holdings..."))

        # Phase 7 – Gemini reasoning (these all happen inside the Gemini call)
        ANALYSIS_STEPS.append(("⚠️", "Gemini: identifying concentration risks and over-weighted positions..."))
        ANALYSIS_STEPS.append(("🏆", "Gemini: ranking top performers and analysing what's driving returns..."))
        ANALYSIS_STEPS.append(("🩸", "Gemini: diagnosing laggards — market conditions, sector headwinds, or entry-price errors?"))
        ANALYSIS_STEPS.append(("💡", "Gemini: drafting specific rebalancing recommendations with allocation percentages..."))
        ANALYSIS_STEPS.append(("🔍", "Gemini: identifying missing sectors and asset classes for a balanced HNI portfolio..."))
        ANALYSIS_STEPS.append(("✍️", "Gemini: writing the overall verdict and the single most important action to take today..."))
        ANALYSIS_STEPS.append(("🎯", "Gemini: self-assessing confidence in the advice based on data completeness..."))
        ANALYSIS_STEPS.append(("📋", "Formatting and finalising the analysis report..."))

        # ~2.2 s per step — for a 62-second response, ~28 steps × 2.2 s ≈ 62 s
        STEP_DURATION = 2.2


        def _collect_full_response(ak, pr):
            """Runs in a background thread: streams Gemini and returns (full_text, model_name)."""
            gen, mname = stream_with_retry(ak, pr)
            return "".join(gen), mname

        try:
            status_box = st.empty()

            # Start the Gemini call in a background thread
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_collect_full_response, api_key, prompt)

                step_idx = 0
                while not future.done():
                    icon, msg = ANALYSIS_STEPS[min(step_idx, len(ANALYSIS_STEPS) - 1)]
                    status_box.markdown(
                        f"""
                        <div style="
                            background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 100%);
                            border-left: 4px solid #3b82f6;
                            border-radius: 8px;
                            padding: 18px 22px;
                            margin: 8px 0;
                            display: flex;
                            align-items: center;
                            gap: 14px;
                        ">
                            <span style="font-size:1.8em; line-height:1">{icon}</span>
                            <div>
                                <div style="color:#93c5fd; font-size:0.75em; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:2px">Gemini AI &bull; Working</div>
                                <div style="color:#f1f5f9; font-size:1em; font-weight:500">{msg}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    # Poll every 0.1s so we react quickly when Gemini finishes
                    for _ in range(int(STEP_DURATION / 0.1)):
                        if future.done():
                            break
                        time.sleep(0.1)
                    step_idx += 1

                full_text, model_name = future.result()  # raises if the thread threw

            status_box.empty()

            # ── Render the analysis section ──
            st.markdown("---")
            # Named anchor so JS can scroll directly to this element
            st.markdown('<div id="ai-analysis"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🤖 Gemini AI Analysis</div>', unsafe_allow_html=True)
            st.caption(f"Model used: {model_name.replace('models/', '')}")

            # ── Strip machine-readable confidence lines before displaying ──
            score_match  = re.search(r"CONFIDENCE_SCORE:\s*(\d+)", full_text)
            reason_match = re.search(r"CONFIDENCE_REASON:\s*(.+?)(?:\n|$)", full_text)
            clean_text   = re.sub(r"\n*CONFIDENCE_SCORE:.*", "", full_text, flags=re.DOTALL).strip()

            st.markdown(clean_text)

            # ── Confidence widget — shown ONCE, below the advice ──
            if score_match:
                ai_confidence_score  = int(score_match.group(1))
                ai_confidence_reason = reason_match.group(1).strip() if reason_match else ""

                st.markdown("---")
                st.markdown('<div class="section-title">🎯 AI Confidence in This Advice</div>', unsafe_allow_html=True)
                st.caption("Gemini's own assessment of how confident it is in the advice above.")

                if ai_confidence_score >= 75:
                    st.metric(label="HIGH CONFIDENCE", value=f"{ai_confidence_score}%")
                    if ai_confidence_reason:
                        st.success(f"💡 {ai_confidence_reason}")
                elif ai_confidence_score >= 50:
                    st.metric(label="MODERATE CONFIDENCE", value=f"{ai_confidence_score}%")
                    if ai_confidence_reason:
                        st.warning(f"💡 {ai_confidence_reason}")
                else:
                    st.metric(label="LOW CONFIDENCE", value=f"{ai_confidence_score}%")
                    if ai_confidence_reason:
                        st.error(f"💡 {ai_confidence_reason}")

        except PermissionDenied:
            st.error("❌ Invalid API key. Please check and try again.")
        except ResourceExhausted:
            st.error("⏳ Rate limit hit on current free quota. Please wait a bit and try again.")
        except GoogleAPICallError as e:
            st.error(f"❌ Google API error: {e}")
        except Exception as e:
            st.error(f"❌ API error: {e}")

        # ── Auto-scroll to #ai-analysis anchor ──
        # This is the most reliable cross-browser approach in Streamlit.
        # We inject a tiny visible link styled as display:none that navigates
        # the browser to the anchor we placed at the analysis heading.
        st.markdown(
            """
            <script>
                // Walk up from the iframe to the parent Streamlit window and
                // scroll the named anchor into view after a short render delay.
                setTimeout(function() {
                    try {
                        var anchor = window.parent.document.getElementById('ai-analysis');
                        if (anchor) {
                            anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        }
                    } catch(e) {}
                }, 300);
            </script>
            """,
            unsafe_allow_html=True,
        )

