#!/usr/bin/env python3
"""
Stock Scanner — Streamlit Web App
Run: streamlit run app.py
"""

import sys
import subprocess

def _ensure(pkg):
    try:
        __import__(pkg.split("[")[0].replace("-", "_"))
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "--break-system-packages"],
            stdout=subprocess.DEVNULL,
        )

for _p in ["streamlit", "yfinance", "pandas", "numpy"]:
    _ensure(_p)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from pandas.io.formats.style import Styler

# ── constants ────────────────────────────────────────────────────────────────

SYMBOLS = ["MU", "MRVL", "WDC", "SNDK", "AMD"]

FOMC_2026 = [
    ("Jul 29–30", "2026-07-29"),
    ("Sep 16–17", "2026-09-16"),
    ("Oct 28–29", "2026-10-28"),
    ("Dec  9–10", "2026-12-09"),
]

# ── helpers ──────────────────────────────────────────────────────────────────

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


@st.cache_data(ttl=600, show_spinner=False)
def fetch_history(symbol: str, period: str = "6mo") -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period)
    if not df.empty:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_earnings(symbol: str):
    """Return next earnings date string or 'N/A'."""
    try:
        cal = yf.Ticker(symbol).calendar
        # calendar is a dict with key 'Earnings Date' → list of Timestamps
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date") or cal.get("earningsDate") or []
        elif isinstance(cal, pd.DataFrame):
            # older yfinance returns a DataFrame
            row = cal.T.get("Earnings Date", pd.Series(dtype=object))
            dates = list(row.dropna())
        else:
            dates = []
        if dates:
            return pd.Timestamp(dates[0]).strftime("%Y-%m-%d")
    except Exception:
        pass
    return "N/A"


def analyze(symbol: str):
    df = fetch_history(symbol)
    if df.empty or len(df) < 50:
        return None

    df = df[["Close", "Volume"]].copy()
    df["MA20"]   = df["Close"].rolling(20).mean()
    df["MA50"]   = df["Close"].rolling(50).mean()
    df["RSI"]    = calc_rsi(df["Close"])
    df["MACD"], df["Signal"] = calc_macd(df["Close"])
    df["VolMA20"] = df["Volume"].rolling(20).mean()

    recent = df.dropna().tail(5)
    if recent.empty:
        return None

    rows = []
    for date, row in recent.iterrows():
        price      = row["Close"]
        idx        = df.index.get_loc(date)
        prev_close = df["Close"].iloc[idx - 1] if idx >= 1 else np.nan
        pct        = ((price - prev_close) / prev_close * 100) if not np.isnan(prev_close) else 0.0

        rsi      = row["RSI"]
        macd_val = row["MACD"]
        sig_val  = row["Signal"]
        vol      = row["Volume"]
        vol_ma20 = row["VolMA20"]
        ma20     = row["MA20"]
        ma50     = row["MA50"]

        vol_flag = "放量 🔥" if vol > vol_ma20 * 1.5 else "正常"
        rsi_flag = "超买 🔴" if rsi > 70 else ("超卖 🟢" if rsi < 30 else "中性")

        if idx >= 1:
            prev_row   = df.iloc[idx - 1]
            prev_cross = prev_row["MACD"] - prev_row["Signal"]
            curr_cross = macd_val - sig_val
            if prev_cross < 0 and curr_cross >= 0:
                macd_flag = "金叉 🟢"
            elif prev_cross > 0 and curr_cross <= 0:
                macd_flag = "死叉 🔴"
            else:
                macd_flag = "金叉上方" if curr_cross > 0 else "死叉下方"
        else:
            macd_flag = "N/A"

        rows.append({
            "日期":        date.strftime("%Y-%m-%d"),
            "收盘价":      round(price, 2),
            "涨跌幅%":     round(pct, 2),
            "成交量(M)":   round(vol / 1e6, 1),
            "成交量状态":  vol_flag,
            "RSI(14)":     round(rsi, 1),
            "RSI状态":     rsi_flag,
            "MACD":        round(macd_val, 3),
            "Signal":      round(sig_val, 3),
            "MACD状态":    macd_flag,
            "MA20位置":    "上方 ▲" if price > ma20 else "下方 ▼",
            "MA50位置":    "上方 ▲" if price > ma50 else "下方 ▼",
        })

    return pd.DataFrame(rows)


def highlight_signals(df: pd.DataFrame) -> Styler:
    """Apply background colors to RSI and MACD columns."""
    def color_row(row):
        styles = [""] * len(row)
        col_names = list(row.index)

        rsi_state_idx  = col_names.index("RSI状态")  if "RSI状态"  in col_names else None
        macd_state_idx = col_names.index("MACD状态") if "MACD状态" in col_names else None
        pct_idx        = col_names.index("涨跌幅%")  if "涨跌幅%"  in col_names else None

        if rsi_state_idx is not None:
            val = str(row["RSI状态"])
            if "超买" in val:
                styles[rsi_state_idx] = "background-color: #ffcccc; color: #900"
            elif "超卖" in val:
                styles[rsi_state_idx] = "background-color: #ccffcc; color: #060"

        if macd_state_idx is not None:
            val = str(row["MACD状态"])
            if "金叉" in val:
                styles[macd_state_idx] = "background-color: #ccffcc; color: #060"
            elif "死叉" in val:
                styles[macd_state_idx] = "background-color: #ffcccc; color: #900"

        if pct_idx is not None:
            pct = row["涨跌幅%"]
            if isinstance(pct, (int, float)):
                if pct > 0:
                    styles[pct_idx] = "color: #0a0; font-weight: bold"
                elif pct < 0:
                    styles[pct_idx] = "color: #c00; font-weight: bold"

        return styles

    return df.style.apply(color_row, axis=1)


# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stock Scanner",
    page_icon="📈",
    layout="wide",
)

# ── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📅 2026 FOMC 会议日期")
    today = datetime.today().date()
    for label, date_str in FOMC_2026:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        days_left = (d - today).days
        if days_left >= 0:
            st.markdown(f"**{label}** — {days_left}天后")
        else:
            st.markdown(f"~~{label}~~")

    st.divider()
    st.header("📋 下次财报日期")
    for sym in SYMBOLS:
        ed = fetch_earnings(sym)
        st.markdown(f"**{sym}** → {ed}")

    st.divider()
    st.caption(f"数据刷新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if st.button("🔄 刷新数据"):
        st.cache_data.clear()
        st.rerun()

# ── main page ─────────────────────────────────────────────────────────────────

st.title("📈 半导体股票扫描器")
st.markdown(f"**扫描标的:** {' · '.join(SYMBOLS)}　　**日期:** {today}")

st.header("技术指标 & 60天走势")

for sym in SYMBOLS:
    with st.spinner(f"正在分析 {sym}..."):
        scan_df = analyze(sym)
        raw = fetch_history(sym, period="6mo")

    st.subheader(sym)

    if scan_df is None:
        st.warning(f"{sym}: 数据不足，跳过")
        st.divider()
        continue

    # ── indicators table ──────────────────────────────────────────────────
    styled = highlight_signals(scan_df)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── 60-day price chart with MA reference lines ────────────────────────
    if not raw.empty and len(raw) >= 50:
        close  = raw["Close"]
        ma20_s = close.rolling(20).mean()
        ma50_s = close.rolling(50).mean()

        current_price = float(close.iloc[-1])
        current_ma20  = float(ma20_s.iloc[-1])
        current_ma50  = float(ma50_s.iloc[-1])

        tail60 = close.tail(60)
        n = len(tail60)
        chart_df = pd.DataFrame(
            {
                "收盘价":   tail60.values,
                "MA20参考": [current_ma20] * n,
                "MA50参考": [current_ma50] * n,
            },
            index=tail60.index,
        )

        col_chart, col_metrics = st.columns([5, 1])
        with col_chart:
            st.caption(f"{sym} — 最近60天收盘价（水平线 = 当前MA20 / MA50）")
            st.line_chart(chart_df, use_container_width=True)
        with col_metrics:
            st.metric("当前价", f"${current_price:.2f}")
            st.metric(
                "MA20",
                f"${current_ma20:.2f}",
                delta=f"{(current_price - current_ma20) / current_ma20 * 100:+.1f}%",
            )
            st.metric(
                "MA50",
                f"${current_ma50:.2f}",
                delta=f"{(current_price - current_ma50) / current_ma50 * 100:+.1f}%",
            )

    st.divider()
