#!/usr/bin/env python3
"""
Stock Scanner — Streamlit Web App
Run: streamlit run app.py
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pandas.io.formats.style import Styler

try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:
    class YFRateLimitError(Exception):
        pass

# Propagate the Streamlit ScriptRunContext into worker threads so that
# @st.cache_data calls work (and don't spam "missing ScriptRunContext").
try:
    from streamlit.runtime.scriptrunner import (
        add_script_run_ctx as _add_ctx,
        get_script_run_ctx as _get_ctx,
    )
except Exception:  # pragma: no cover — very old/new Streamlit
    _add_ctx = None

    def _get_ctx():
        return None

# ── constants ────────────────────────────────────────────────────────────────

SYMBOLS = ["MU", "MRVL", "WDC", "SNDK", "AMD", "ASML"]

BLUE_CHIPS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "TSM", "ORCL",
    "ASML", "AMD", "MU", "QCOM", "TXN", "ARM", "MRVL", "AMAT", "LRCX", "KLAC",
    "JPM", "BAC", "GS", "MS", "WFC", "BRK-B", "V", "MA",
    "UNH", "LLY", "JNJ", "ABBV",
    "XOM", "CVX",
    "WMT", "COST", "HD",
    "GE", "CAT", "RTX", "LMT", "GEV", "VRT",
    "NFLX", "DIS",
    "WDC", "SNDK", "COIN", "PLTR",
]

SEC_UA = "StockScanner windfocus@gmail.com"

# Concurrency / retry tuning. max_workers kept low to stay under Yahoo limits.
MAX_WORKERS = 5
RETRY_ROUNDS = 3                 # total attempts (1 initial + retries), max 3
BACKOFF = (2, 4, 8)             # exponential back-off (seconds) between rounds

ITEM_DESC = {
    "1.01": "签订重大协议",
    "1.02": "终止重大协议",
    "1.03": "破产或接管",
    "1.04": "矿山安全事项",
    "1.05": "重大网络安全事件",
    "2.01": "完成重大资产收购或处置",
    "2.02": "财报业绩披露",
    "2.03": "创设直接金融义务",
    "2.04": "触发加速或增加金融义务",
    "2.05": "裁员或退出计划",
    "2.06": "资产减值",
    "3.01": "退市或转板通知",
    "3.02": "未注册股权销售",
    "3.03": "修改股东权利",
    "4.01": "更换会计师",
    "4.02": "会计师非依赖声明",
    "5.01": "控制权变更",
    "5.02": "高管离职或任命",
    "5.03": "修订公司章程",
    "5.07": "股东提名通知",
    "7.01": "Regulation FD 信息披露",
    "8.01": "其他重大事件",
    "9.01": "财务报表及附件",
}

FOMC_2026 = [
    ("Jul 29–30", "2026-07-29"),
    ("Sep 16–17", "2026-09-16"),
    ("Oct 28–29", "2026-10-28"),
    ("Dec  9–10", "2026-12-09"),
]

# ── data-layer exceptions ─────────────────────────────────────────────────────


class DataUnavailable(Exception):
    """Terminal failure fetching data (network / parse error)."""


class RateLimited(DataUnavailable):
    """Yahoo Finance throttled the request — worth retrying with back-off."""


# ── concurrency helpers ───────────────────────────────────────────────────────


def _parallel_map(fn, items, max_workers=MAX_WORKERS, progress=None, label="加载中"):
    """Run ``fn(item)`` across a thread pool.

    Returns ``{item: (result, exception_or_None)}``. Never raises: any
    exception from ``fn`` is captured so a single failing symbol cannot take
    down the rest of the batch. Updates ``progress`` (a st.progress handle)
    as futures complete.
    """
    items = list(items)
    total = len(items)
    results = {}
    if total == 0:
        return results

    ctx = _get_ctx()

    def _wrapped(item):
        if _add_ctx is not None and ctx is not None:
            _add_ctx(threading.current_thread(), ctx)
        try:
            return item, fn(item), None
        except Exception as exc:  # noqa: BLE001 — captured per-item on purpose
            return item, None, exc

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_wrapped, it) for it in items]
        for fut in as_completed(futures):
            item, res, err = fut.result()
            results[item] = (res, err)
            done += 1
            if progress is not None:
                progress.progress(done / total, text=f"{label} {done}/{total}")
    return results


# ── perf tracking (shown in sidebar) ──────────────────────────────────────────


def _add_fetch(secs: float):
    st.session_state["_fetch_secs"] = st.session_state.get("_fetch_secs", 0.0) + secs
    st.session_state["_last_refresh"] = datetime.now().strftime("%H:%M")


# ── indicator helpers ─────────────────────────────────────────────────────────


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


def calc_kdj(df, n=9, m=3):
    low_n  = df["Low"].rolling(n, min_periods=1).min()
    high_n = df["High"].rolling(n, min_periods=1).max()
    rsv = (df["Close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    K = rsv.ewm(com=m - 1, min_periods=1, adjust=False).mean()
    D = K.ewm(com=m - 1, min_periods=1, adjust=False).mean()
    J = 3 * K - 2 * D
    return K, D, J


# ── data fetchers (cached) ─────────────────────────────────────────────────────
# NOTE: on rate-limit / transient failure these RAISE rather than returning an
# empty value. st.cache_data does not cache exceptions, so a throttled request
# is never poisoned into the cache — the next attempt re-fetches cleanly. This
# is what prevents one rate-limit storm from making *all* symbols read
# "数据不足" for the rest of the TTL window.


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(symbol: str, period: str = "6mo") -> pd.DataFrame:
    """Daily OHLCV history. TTL = 15 min. Raises on failure (not cached)."""
    try:
        df = yf.Ticker(symbol).history(period=period)
    except YFRateLimitError as exc:
        raise RateLimited(f"{symbol}: rate limited") from exc
    except Exception as exc:  # noqa: BLE001
        raise DataUnavailable(f"{symbol}: {exc}") from exc
    if df.empty:
        # Empty payloads are usually silent throttling — treat as retryable.
        raise RateLimited(f"{symbol}: empty history")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_gamma_wall(symbol: str):
    """Option-chain derived PCR + gamma walls. TTL = 30 min.

    Returns a dict on success, ``None`` when the symbol genuinely has no
    options. Raises RateLimited/DataUnavailable on failure (not cached).
    """
    import math
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            return None
        expiry = expirations[0]
        chain  = ticker.option_chain(expiry)

        try:
            price = float(ticker.fast_info.last_price)
        except Exception:
            price = float(ticker.history(period="1d")["Close"].iloc[-1])
        if not price or price <= 0:
            return None

        call_oi_series = chain.calls["openInterest"].dropna()
        put_oi_series  = chain.puts["openInterest"].dropna()

        call_oi_total = call_oi_series.sum(min_count=1)
        put_oi_total  = put_oi_series.sum(min_count=1)

        if (math.isnan(float(call_oi_total)) or call_oi_total == 0
                or math.isnan(float(put_oi_total))):
            pcr = None
        else:
            pcr = round(float(put_oi_total) / float(call_oi_total), 3)

        calls_all = chain.calls[["strike", "openInterest"]].rename(columns={"openInterest": "call_oi"})
        puts_all  = chain.puts[["strike", "openInterest"]].rename(columns={"openInterest": "put_oi"})

        if math.isnan(price) or price <= 0:
            return None
        lo, hi = price * 0.75, price * 1.25
        calls = calls_all[(calls_all["strike"] >= lo) & (calls_all["strike"] <= hi)].copy()
        puts  = puts_all[(puts_all["strike"] >= lo) & (puts_all["strike"] <= hi)].copy()

        call_above = calls[calls["strike"] > price].nlargest(3, "call_oi")
        put_below  = puts[puts["strike"] < price].nlargest(3, "put_oi")

        nearest_call = call_above.sort_values("strike")["strike"].iloc[0] if not call_above.empty else None
        nearest_put  = put_below.sort_values("strike", ascending=False)["strike"].iloc[0] if not put_below.empty else None

        merged = (
            pd.merge(calls, puts, on="strike", how="outer")
            .fillna(0)
            .sort_values("strike")
            .reset_index(drop=True)
        )

        return {
            "expiry":       expiry,
            "price":        price,
            "merged":       merged,
            "call_walls":   sorted(call_above["strike"].tolist()),
            "put_walls":    sorted(put_below["strike"].tolist(), reverse=True),
            "nearest_call": nearest_call,
            "nearest_put":  nearest_put,
            "pcr":          pcr,
        }
    except YFRateLimitError as exc:
        raise RateLimited(f"{symbol}: options rate limited") from exc
    except Exception as exc:  # noqa: BLE001
        raise DataUnavailable(f"{symbol}: options {exc}") from exc


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_earnings(symbol: str):
    """Next earnings date. TTL = 60 min. Raises only on rate limit."""
    try:
        cal = yf.Ticker(symbol).calendar
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date") or cal.get("earningsDate") or []
        elif isinstance(cal, pd.DataFrame):
            row = cal.T.get("Earnings Date", pd.Series(dtype=object))
            dates = list(row.dropna())
        else:
            dates = []
        if dates:
            return pd.Timestamp(dates[0]).strftime("%Y-%m-%d")
        return "N/A"
    except YFRateLimitError as exc:
        raise RateLimited(f"{symbol}: earnings rate limited") from exc
    except Exception:
        return "N/A"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cik_map() -> dict:
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": SEC_UA},
            timeout=10,
        )
        return {v["ticker"].upper(): int(v["cik_str"]) for v in r.json().values()}
    except Exception:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_8k_filings(symbol: str, cik: int, days: int = 10) -> list[dict]:
    cutoff = (datetime.today() - timedelta(days=days)).date()
    results = []
    try:
        r = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
            headers={"User-Agent": SEC_UA},
            timeout=10,
        )
        recent = r.json().get("filings", {}).get("recent", {})
        forms      = recent.get("form", [])
        dates      = recent.get("filingDate", [])
        items_list = recent.get("items", [])

        for form, date_str, raw_items in zip(forms, dates, items_list):
            if form not in ("8-K", "8-K/A", "6-K", "6-K/A"):
                continue
            if datetime.strptime(date_str, "%Y-%m-%d").date() < cutoff:
                break
            items = [
                x.strip()
                for x in str(raw_items).split(",")
                if x.strip() and x.strip().lower() != "nan"
            ]
            results.append({
                "symbol":  symbol,
                "date":    date_str,
                "form":    form,
                "items":   items,
            })
    except Exception:
        pass
    return results


# ── analysis ──────────────────────────────────────────────────────────────────


def analyze(symbol: str, df: pd.DataFrame | None = None):
    """Build the 5-row technical table. ``df`` may be a pre-fetched history.

    Returns a DataFrame, or ``None`` when there is genuinely too little
    history. Propagates DataUnavailable if a fetch is needed and fails.
    """
    if df is None:
        df = fetch_history(symbol)
    if df is None or df.empty or len(df) < 50:
        return None

    df = df[["Close", "High", "Low", "Volume"]].copy()
    df["MA20"]    = df["Close"].rolling(20).mean()
    df["MA50"]    = df["Close"].rolling(50).mean()
    df["RSI"]     = calc_rsi(df["Close"])
    df["MACD"], df["Signal"] = calc_macd(df["Close"])
    df["VolMA20"] = df["Volume"].rolling(20).mean()
    df["K"], df["D"], df["J"] = calc_kdj(df)

    bb_mid          = df["Close"].rolling(20).mean()
    bb_std          = df["Close"].rolling(20).std()
    df["BB_Width"]  = 4 * bb_std / bb_mid * 100
    df["BB_W_MA60"] = df["BB_Width"].rolling(60, min_periods=30).mean()

    recent = df.dropna().tail(5)
    if recent.empty:
        return None

    rows = []
    for date, row in recent.iterrows():
        price      = row["Close"]
        idx        = df.index.get_loc(date)
        prev_close = df["Close"].iloc[idx - 1] if idx >= 1 else np.nan
        pct        = ((price - prev_close) / prev_close * 100) if not np.isnan(prev_close) else 0.0

        vol      = row["Volume"]
        vol_ma20 = row["VolMA20"]
        ma20     = row["MA20"]
        ma50     = row["MA50"]
        rsi      = row["RSI"]
        macd_val = row["MACD"]
        sig_val  = row["Signal"]
        j_val    = row["J"]
        bb_w     = row["BB_Width"]
        bb_avg   = row["BB_W_MA60"]

        vol_flag = "放量 🔥" if vol > vol_ma20 * 1.5 else "正常"
        rsi_flag = "超买 🔴" if rsi > 70 else ("超卖 🟢" if rsi < 30 else "中性")
        j_flag   = "黄金坑 🟢" if j_val < 20 else ("超买 🔴" if j_val > 100 else "正常")
        bb_flag  = "蓄力 🔋" if (not np.isnan(bb_avg) and bb_w < bb_avg * 0.70) else "正常"

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
            "日期":       date.strftime("%Y-%m-%d"),
            "收盘价":     round(price, 2),
            "涨跌幅%":    round(pct, 2),
            "成交量(M)":  round(vol / 1e6, 1),
            "成交量状态": vol_flag,
            "RSI(14)":    round(rsi, 1),
            "RSI状态":    rsi_flag,
            "MACD":       round(macd_val, 3),
            "Signal":     round(sig_val, 3),
            "MACD状态":   macd_flag,
            "MA20位置":   "上方 ▲" if price > ma20 else "下方 ▼",
            "MA50位置":   "上方 ▲" if price > ma50 else "下方 ▼",
            "J值":        round(j_val, 1),
            "J状态":      j_flag,
            "BB挤压":     bb_flag,
        })

    return pd.DataFrame(rows)


def highlight_signals(df: pd.DataFrame) -> Styler:
    def color_row(row):
        styles = [""] * len(row)
        col_names = list(row.index)

        rsi_state_idx  = col_names.index("RSI状态")  if "RSI状态"  in col_names else None
        macd_state_idx = col_names.index("MACD状态") if "MACD状态" in col_names else None
        pct_idx        = col_names.index("涨跌幅%")  if "涨跌幅%"  in col_names else None
        j_val_idx      = col_names.index("J值")      if "J值"      in col_names else None
        bb_idx         = col_names.index("BB挤压")   if "BB挤压"   in col_names else None

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

        if j_val_idx is not None:
            j = row["J值"]
            if isinstance(j, (int, float)):
                if j < 20:
                    styles[j_val_idx] = "background-color: #ccffcc; color: #060; font-weight: bold"
                elif j > 100:
                    styles[j_val_idx] = "background-color: #ffcccc; color: #900; font-weight: bold"

        if bb_idx is not None:
            if "蓄力" in str(row["BB挤压"]):
                styles[bb_idx] = "background-color: #fff3cd; color: #856404; font-weight: bold"

        return styles

    return df.style.apply(color_row, axis=1)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_macro_sina() -> dict:
    """Fetch macro indicators from Sina Finance (5-min cache)."""
    try:
        url = "https://hq.sinajs.cn/list=hf_GC,hf_SI,hf_HG,hf_DX,hf_VIX,gb_%24tnx"
        headers = {
            "Referer": "https://finance.sina.com.cn/",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        r = requests.get(url, headers=headers, timeout=10)
        r.encoding = "gbk"
        result = {}
        for line in r.text.strip().split("\n"):
            if '="' not in line:
                continue
            key_part, _, val_part = line.partition('="')
            key = key_part.replace("var hq_str_", "").strip()
            key = key.replace("%24", "$")  # normalize URL-encoded $
            val = val_part.rstrip("\n\r").rstrip(";").rstrip('"')
            fields = val.split(",")
            if not fields or not fields[0].strip():
                continue
            try:
                if key.startswith("hf_") and len(fields) >= 5:
                    current = float(fields[0])
                    prev    = float(fields[4])
                    change  = current - prev
                    result[key] = {
                        "current":    current,
                        "prev":       prev,
                        "change":     round(change, 4),
                        "change_pct": round(change / prev * 100 if prev else 0.0, 2),
                        "high":       float(fields[1]),
                        "low":        float(fields[2]),
                    }
                elif key.startswith("gb_") and len(fields) >= 5:
                    result[key] = {
                        "current":    float(fields[0]),
                        "high":       float(fields[1]),
                        "low":        float(fields[2]),
                        "change":     float(fields[4]),
                        "change_pct": float(fields[5]) if len(fields) > 5 else 0.0,
                    }
            except (ValueError, IndexError):
                result[key] = None
        return result
    except Exception:
        return {}


def scan_one_golden(symbol: str):
    """Return metrics dict if symbol passes golden-pit criteria, else None.

    Lets RateLimited / DataUnavailable propagate so the orchestrator can
    retry throttled symbols instead of silently dropping them.
    """
    df = fetch_history(symbol)
    if df.empty or len(df) < 55:
        return None
    df = df[["Close", "High", "Low", "Volume"]].copy()
    df["MA50"]   = df["Close"].rolling(50).mean()
    df["RSI"]    = calc_rsi(df["Close"])
    df["VolMA20"]= df["Volume"].rolling(20).mean()
    _, _, j_series = calc_kdj(df)
    df["J"] = j_series
    clean = df.dropna()
    if len(clean) < 6:
        return None
    latest  = clean.iloc[-1]
    price   = latest["Close"]
    j_val   = latest["J"]
    if j_val >= 25 or price <= latest["MA50"]:
        return None
    price_5d = clean["Close"].iloc[-6]
    ret_5d   = (price - price_5d) / price_5d * 100
    if ret_5d >= -3.0:
        return None
    vol_status = "放量 🔥" if latest["Volume"] > latest["VolMA20"] * 1.5 else "正常"
    return {
        "代码":       symbol,
        "现价":       round(price, 2),
        "J值":        round(j_val, 1),
        "近5日涨跌%": round(ret_5d, 2),
        "RSI":        round(latest["RSI"], 1),
        "成交量状态": vol_status,
    }


def highlight_golden_pit(df: pd.DataFrame) -> Styler:
    def color_row(row):
        if row["J值"] < 15:
            return ["background-color: #ffcccc; color: #900; font-weight: bold"] * len(row)
        return [""] * len(row)
    return df.style.apply(color_row, axis=1)


# ── orchestration (concurrent fetch + rate-limit retry rounds) ─────────────────


def _load_bundle(symbol: str, want_options: bool) -> dict:
    """Fetch everything Tab1 needs for one symbol, isolating each failure."""
    b = {
        "hist": None, "hist_err": None,
        "gamma": None, "gamma_err": None,
        "earnings": "N/A", "rate_limited": False,
    }
    try:
        b["hist"] = fetch_history(symbol)
    except RateLimited:
        b["rate_limited"] = True
        b["hist_err"] = "ratelimit"
    except DataUnavailable as exc:
        b["hist_err"] = str(exc)

    if want_options:
        try:
            b["gamma"] = fetch_gamma_wall(symbol)
        except RateLimited:
            b["rate_limited"] = True
            b["gamma_err"] = "ratelimit"
        except DataUnavailable as exc:
            b["gamma_err"] = str(exc)

    try:
        b["earnings"] = fetch_earnings(symbol)
    except (RateLimited, DataUnavailable):
        b["rate_limited"] = True
        b["earnings"] = "加载中"
    return b


def warm_symbols(symbols, want_options=True, progress=None, status=None,
                 max_workers=MAX_WORKERS) -> dict:
    """Concurrently load Tab1 bundles, retrying only the throttled symbols.

    Retries up to RETRY_ROUNDS times with exponential back-off. Anything that
    already succeeded is served from cache on the retry round, so retries are
    cheap. Shows a live "限速中，将在X秒后重试" message via ``status``.
    """
    bundles = {}
    pending = list(symbols)
    for attempt in range(RETRY_ROUNDS):
        out = _parallel_map(
            lambda s: _load_bundle(s, want_options),
            pending, max_workers=max_workers, progress=progress,
            label=f"加载技术数据 (第{attempt + 1}轮)",
        )
        retry = []
        for sym, (res, err) in out.items():
            if err is not None or res is None:
                bundles[sym] = {
                    "hist": None, "hist_err": str(err), "gamma": None,
                    "gamma_err": None, "earnings": "N/A", "rate_limited": False,
                }
                continue
            bundles[sym] = res
            if res.get("rate_limited"):
                retry.append(sym)
        if not retry or attempt == RETRY_ROUNDS - 1:
            break
        delay = BACKOFF[attempt]
        if status is not None:
            status.warning(
                f"⚠️ Yahoo Finance 限速中，将在 {delay} 秒后重试"
                f"（{len(retry)} 只待重试）..."
            )
        time.sleep(delay)
        pending = retry
    if status is not None:
        status.empty()
    return bundles


def run_golden_scan(symbols, progress=None, status=None, max_workers=MAX_WORKERS):
    """Concurrent golden-pit scan with rate-limit retry rounds."""
    found = []
    pending = list(symbols)
    for attempt in range(RETRY_ROUNDS):
        out = _parallel_map(
            scan_one_golden, pending, max_workers=max_workers, progress=progress,
            label=f"黄金坑扫描 (第{attempt + 1}轮)",
        )
        retry = []
        for sym, (res, err) in out.items():
            if isinstance(err, RateLimited):
                retry.append(sym)
            elif err is None and res:
                found.append(res)
            # else: no match, or non-retryable DataUnavailable → skip
        if not retry or attempt == RETRY_ROUNDS - 1:
            break
        delay = BACKOFF[attempt]
        if status is not None:
            status.warning(
                f"⚠️ Yahoo Finance 限速中，将在 {delay} 秒后重试"
                f"（{len(retry)} 只待重试）..."
            )
        time.sleep(delay)
        pending = retry
    if status is not None:
        status.empty()
    return found


# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stock Scanner",
    page_icon="📈",
    layout="wide",
)

# Per-run fetch timer reset; last-refresh persists across reruns.
st.session_state["_fetch_secs"] = 0.0
st.session_state.setdefault("_last_refresh", "—")

today = datetime.today().date()

# ── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📅 2026 FOMC 会议日期")
    for label, date_str in FOMC_2026:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        days_left = (d - today).days
        if days_left >= 0:
            st.markdown(f"**{label}** — {days_left}天后")
        else:
            st.markdown(f"~~{label}~~")

    st.divider()
    st.header("📋 下次财报日期")
    _t0 = time.perf_counter()
    _ed_map = _parallel_map(fetch_earnings, SYMBOLS, max_workers=MAX_WORKERS)
    _add_fetch(time.perf_counter() - _t0)
    for sym in SYMBOLS:
        _res, _err = _ed_map.get(sym, (None, None))
        ed = _res if (_err is None and _res) else "加载中"
        st.markdown(f"**{sym}** → {ed}")

    st.divider()
    st.caption(f"数据刷新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if st.button("🔄 刷新数据"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    perf_ph = st.empty()  # filled at end of script with refresh time + duration

# ── main page ─────────────────────────────────────────────────────────────────

st.title("📈 股票扫描器")
st.markdown(f"**日期:** {today}")
st.info("💡 如遇数据加载失败，请等待30秒后点击侧边栏「刷新数据」按钮重试。")

tab1, tab2, tab3 = st.tabs(["📊 技术扫描", "🎯 黄金坑", "🌍 宏观"])

# ══════════════════════════════════════════════════════════════════════════════
# Tab1 — 技术扫描
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown(f"**扫描标的:** {' · '.join(SYMBOLS)}")

    st.header("📋 重大事件 (8-K / 6-K) — 最近10天")

    with st.spinner("正在从 SEC EDGAR 抓取最新文件..."):
        cik_map = fetch_cik_map()
        all_filings = []
        for sym in SYMBOLS:
            cik = cik_map.get(sym)
            if cik:
                all_filings.extend(fetch_8k_filings(sym, cik, days=10))

    if all_filings:
        all_filings.sort(key=lambda x: x["date"], reverse=True)
        rows_8k = []
        for f in all_filings:
            if f["items"]:
                for item in f["items"]:
                    rows_8k.append({
                        "公司":     f["symbol"],
                        "提交日期": f["date"],
                        "文件类型": f["form"],
                        "条目":     item,
                        "说明":     ITEM_DESC.get(item, "其他事项"),
                    })
            else:
                rows_8k.append({
                    "公司":     f["symbol"],
                    "提交日期": f["date"],
                    "文件类型": f["form"],
                    "条目":     "—",
                    "说明":     "（条目信息不适用）",
                })
        st.dataframe(pd.DataFrame(rows_8k), use_container_width=True, hide_index=True)
    else:
        st.info("过去10天内，以上股票均无 8-K / 6-K 提交记录。")

    st.divider()
    st.header("技术指标 & 60天走势")

    # ── concurrent load of all 6 symbols (history + options + earnings) ──────
    _prog   = st.progress(0.0, text="加载技术指标数据...")
    _status = st.empty()
    _t0 = time.perf_counter()
    bundles = warm_symbols(SYMBOLS, want_options=True, progress=_prog,
                           status=_status, max_workers=MAX_WORKERS)
    _prog.empty()
    _add_fetch(time.perf_counter() - _t0)

    for sym in SYMBOLS:
        b = bundles.get(sym, {})
        st.subheader(sym)

        if b.get("hist") is None:
            st.warning("⚠️ 该股票数据暂不可用（Yahoo Finance 限速或网络问题），其他股票不受影响。")
            st.divider()
            continue

        scan_df = analyze(sym, b["hist"])
        if scan_df is None:
            st.warning(f"{sym}: 数据不足，跳过")
            st.divider()
            continue

        gw         = b.get("gamma")
        gw_pending = b.get("gamma_err") is not None  # failed/throttled, not "no options"
        raw        = b["hist"]

        tbl_col, pcr_col = st.columns([8, 2])
        with tbl_col:
            st.dataframe(highlight_signals(scan_df), use_container_width=True, hide_index=True)
        with pcr_col:
            if gw and gw["pcr"] is not None:
                pcr = gw["pcr"]
                if pcr > 1.2:
                    pcr_label, pcr_color = "看跌情绪重 🔴", "#cc0000"
                elif pcr < 0.7:
                    pcr_label, pcr_color = "看涨情绪重 🟢", "#006600"
                else:
                    pcr_label, pcr_color = "情绪中性 ⚪", "#555555"
                st.markdown(
                    f"""<div style="border:2px solid {pcr_color}; border-radius:10px;
                        padding:14px; text-align:center; margin-top:6px;">
                        <div style="font-size:11px; color:#888; margin-bottom:4px;">期权情绪 (PCR)</div>
                        <div style="font-size:30px; font-weight:bold; color:{pcr_color};">{pcr}</div>
                        <div style="font-size:13px; color:{pcr_color}; margin-top:4px;">{pcr_label}</div>
                        <div style="font-size:10px; color:#aaa; margin-top:6px;">最近到期日数据</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                if gw_pending:
                    msg = "数据加载中..."
                elif gw is None:
                    msg = "无期权数据"
                else:
                    msg = "数据不足"
                st.markdown(
                    f"""<div style="border:1px solid #ddd; border-radius:10px;
                        padding:14px; text-align:center; margin-top:6px; color:#aaa;">
                        <div style="font-size:11px;">期权情绪 (PCR)</div>
                        <div style="font-size:13px; margin-top:6px;">{msg}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        if gw:
            price = gw["price"]
            nc  = f"${gw['nearest_call']:.2f}" if gw["nearest_call"] else "—"
            np_ = f"${gw['nearest_put']:.2f}"  if gw["nearest_put"]  else "—"
            cw  = " / ".join(f"${s:.2f}" for s in gw["call_walls"]) or "—"
            pw  = " / ".join(f"${s:.2f}" for s in gw["put_walls"])  or "—"

            try:
                merged = gw["merged"]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=merged["strike"], y=merged["call_oi"],
                    name="Call OI（阻力）",
                    marker_color="rgba(220,50,50,0.75)",
                ))
                fig.add_trace(go.Bar(
                    x=merged["strike"], y=merged["put_oi"],
                    name="Put OI（支撑）",
                    marker_color="rgba(50,180,50,0.75)",
                ))
                fig.add_vline(
                    x=price, line_dash="dash", line_color="#ff9900", line_width=2,
                    annotation_text=f"  当前 ${price:.2f}",
                    annotation_font_color="#ff9900",
                    annotation_font_size=12,
                )
                for s in gw["call_walls"]:
                    fig.add_vline(x=s, line_dash="dot", line_color="rgba(220,50,50,0.35)", line_width=1)
                for s in gw["put_walls"]:
                    fig.add_vline(x=s, line_dash="dot", line_color="rgba(50,180,50,0.35)", line_width=1)
                fig.update_layout(
                    title=dict(text=f"Gamma 墙分析 — {sym}  （到期日: {gw['expiry']}）", font_size=14),
                    barmode="group",
                    height=320,
                    margin=dict(l=0, r=0, t=45, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
                    xaxis=dict(title="行权价 ($)", showgrid=True, gridcolor="#eee"),
                    yaxis=dict(title="未平仓量", showgrid=True, gridcolor="#eee"),
                    plot_bgcolor="white",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"图表加载失败（{e}）")

            st.markdown(
                f"📍 当前价 **${price:.2f}**　│　"
                f"🔴 上方最近阻力 (Call Wall) **{nc}**　│　"
                f"🟢 下方最近支撑 (Put Wall) **{np_}**"
            )
            st.caption(f"Call Walls Top3: {cw}　　　Put Walls Top3: {pw}")
        elif gw_pending:
            st.caption("期权 Gamma 墙数据加载中...（限速或网络问题，技术指标不受影响）")

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


# ══════════════════════════════════════════════════════════════════════════════
# Tab2 — 黄金坑
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("🎯 黄金坑扫描")
    st.markdown(
        "**筛选条件:** J值 < 25 　·　 价格在MA50上方 　·　 近5日涨跌幅 < −3%  \n"
        f"**股票池:** {len(BLUE_CHIPS)} 只大市值蓝筹（纳指100 + 标普500）"
    )

    if st.button("🚀 开始扫描", key="btn_golden"):
        prog   = st.progress(0.0, text="初始化...")
        status = st.empty()
        t0 = time.perf_counter()
        found = run_golden_scan(BLUE_CHIPS, progress=prog, status=status,
                                max_workers=MAX_WORKERS)
        prog.empty()
        _add_fetch(time.perf_counter() - t0)
        st.session_state["pit_results"]   = found
        st.session_state["pit_scan_time"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    if "pit_results" in st.session_state:
        pit_results = st.session_state["pit_results"]
        scan_time   = st.session_state.get("pit_scan_time", "")
        if pit_results:
            result_df = (
                pd.DataFrame(pit_results)
                .sort_values("J值")
                .reset_index(drop=True)
            )
            st.caption(
                f"扫描时间: {scan_time}　　"
                f"找到 **{len(pit_results)}** 只符合条件　　"
                f"🔴 红色行 = J值<15 强烈信号"
            )
            styled = highlight_golden_pit(result_df).format({
                "现价":       "${:.2f}",
                "J值":        "{:.1f}",
                "近5日涨跌%": "{:.2f}%",
                "RSI":        "{:.1f}",
            })
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.info(
                f"扫描时间: {scan_time}\n\n"
                "当前没有符合黄金坑条件的股票（J<25 且 价格在MA50上方 且 近5日涨跌<−3%）"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Tab3 — 宏观
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("🌍 宏观市场指标")

    if st.button("🔄 刷新宏观数据", key="btn_macro"):
        fetch_macro_sina.clear()
        st.rerun()

    _prog_m = st.progress(0.0, text="正在从新浪财经获取宏观数据...")
    _t0 = time.perf_counter()
    macro = fetch_macro_sina()
    _prog_m.progress(1.0, text="完成")
    _add_fetch(time.perf_counter() - _t0)
    _prog_m.empty()

    if not macro:
        st.error("无法获取宏观数据，请检查网络或稍后重试。")
    else:
        def _m(key: str) -> dict:
            return macro.get(key) or {}

        gold   = _m("hf_GC")
        silver = _m("hf_SI")
        copper = _m("hf_HG")
        dxy    = _m("hf_DX")
        vix    = _m("hf_VIX")
        tnx    = _m("gb_$tnx")

        # ── Row 1: Commodities + USD ──────────────────────────────────────────
        st.subheader("大宗商品 & 美元指数")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            if gold:
                st.metric(
                    "🥇 黄金 ($/oz)",
                    f"${gold['current']:.2f}",
                    delta=f"{gold['change']:+.2f}  ({gold['change_pct']:+.2f}%)",
                )
            else:
                st.metric("🥇 黄金", "—")

        with c2:
            if silver:
                st.metric(
                    "🥈 白银 ($/oz)",
                    f"${silver['current']:.3f}",
                    delta=f"{silver['change']:+.3f}  ({silver['change_pct']:+.2f}%)",
                )
            else:
                st.metric("🥈 白银", "—")

        with c3:
            if copper:
                st.metric(
                    "🔶 铜 ($/lb)",
                    f"${copper['current']:.4f}",
                    delta=f"{copper['change']:+.4f}  ({copper['change_pct']:+.2f}%)",
                )
            else:
                st.metric("🔶 铜", "—")

        with c4:
            if dxy:
                st.metric(
                    "💵 美元指数 (DXY)",
                    f"{dxy['current']:.3f}",
                    delta=f"{dxy['change']:+.3f}  ({dxy['change_pct']:+.2f}%)",
                )
            else:
                st.metric("💵 美元指数", "—")

        st.divider()

        # ── Row 2: VIX + Treasury + Derived ratios ────────────────────────────
        st.subheader("风险情绪 & 衍生比率")
        c5, c6, c7, c8 = st.columns(4)

        with c5:
            if vix:
                vix_val   = vix["current"]
                vix_delta = f"{vix['change']:+.3f}  ({vix['change_pct']:+.2f}%)"
                if vix_val > 20:
                    st.markdown(
                        f'<div style="background:#ffcccc;padding:14px;border-radius:8px;'
                        f'border:2px solid #cc0000;text-align:center">'
                        f'<div style="font-size:12px;color:#666">⚠️ VIX 恐慌指数</div>'
                        f'<div style="font-size:32px;font-weight:bold;color:#cc0000">{vix_val:.2f}</div>'
                        f'<div style="font-size:13px;color:#cc0000;font-weight:bold">市场恐慌 🔴</div>'
                        f'<div style="font-size:11px;color:#888;margin-top:4px">{vix_delta}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                elif vix_val < 15:
                    st.markdown(
                        f'<div style="background:#ccffcc;padding:14px;border-radius:8px;'
                        f'border:2px solid #006600;text-align:center">'
                        f'<div style="font-size:12px;color:#666">✅ VIX 恐慌指数</div>'
                        f'<div style="font-size:32px;font-weight:bold;color:#006600">{vix_val:.2f}</div>'
                        f'<div style="font-size:13px;color:#006600;font-weight:bold">市场平静 🟢</div>'
                        f'<div style="font-size:11px;color:#888;margin-top:4px">{vix_delta}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.metric("⚡ VIX 恐慌指数", f"{vix_val:.2f}", delta=vix_delta)
            else:
                st.metric("⚡ VIX", "—")

        with c6:
            if tnx:
                st.metric(
                    "📊 10年美债收益率",
                    f"{tnx['current']:.3f}%",
                    delta=f"{tnx['change']:+.3f}%  ({tnx['change_pct']:+.2f}%)",
                )
            else:
                st.metric("📊 10年美债", "—")

        with c7:
            if gold and silver and silver.get("current", 0) > 0:
                gs_ratio = gold["current"] / silver["current"]
                st.metric(
                    "⚖️ 金银比",
                    f"{gs_ratio:.1f}",
                    help="Gold/Silver ratio。历史均值约70-80，比值高 = 白银相对低估",
                )
            else:
                st.metric("⚖️ 金银比", "—")

        with c8:
            if copper and gold and gold.get("current", 0) > 0:
                cg_ratio = copper["current"] / gold["current"]
                st.metric(
                    "🔶/🥇 铜金比",
                    f"{cg_ratio:.5f}",
                    help="Copper($/lb) ÷ Gold($/oz)。比值上升通常预示经济扩张",
                )
            else:
                st.metric("🔶/🥇 铜金比", "—")

        # ── Gold detail box ───────────────────────────────────────────────────
        if gold:
            st.divider()
            st.subheader("🥇 黄金详情（现货参考）")
            g1, g2, g3, g4 = st.columns(4)
            with g1:
                st.metric("当前价", f"${gold['current']:.2f}")
            with g2:
                arrow = "▲" if gold["change"] >= 0 else "▼"
                st.metric("今日涨跌", f"{arrow} ${abs(gold['change']):.2f}")
            with g3:
                st.metric("今日涨跌%", f"{gold['change_pct']:+.2f}%")
            with g4:
                st.metric("日内区间", f"${gold['low']:.2f} – ${gold['high']:.2f}")


# ── sidebar perf line (filled after all tabs have fetched this run) ────────────

perf_ph.caption(
    f"上次刷新：{st.session_state.get('_last_refresh', '—')} | "
    f"数据获取耗时：{st.session_state.get('_fetch_secs', 0.0):.1f}秒"
)
