#!/usr/bin/env python3
"""
Stock Scanner — Streamlit Web App
Run: streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pandas.io.formats.style import Styler

# ── constants ────────────────────────────────────────────────────────────────

SYMBOLS = ["MU", "MRVL", "WDC", "SNDK", "AMD", "ASML"]

SEC_UA = "StockScanner windfocus@gmail.com"

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


def calc_kdj(df, n=9, m=3):
    low_n  = df["Low"].rolling(n, min_periods=1).min()
    high_n = df["High"].rolling(n, min_periods=1).max()
    rsv = (df["Close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    K = rsv.ewm(com=m - 1, min_periods=1, adjust=False).mean()
    D = K.ewm(com=m - 1, min_periods=1, adjust=False).mean()
    J = 3 * K - 2 * D
    return K, D, J


@st.cache_data(ttl=600, show_spinner=False)
def fetch_history(symbol: str, period: str = "6mo") -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period)
    if not df.empty:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_gamma_wall(symbol: str):
    """Fetch nearest-expiry option chain; return gamma wall dict or None."""
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            return None
        expiry = expirations[0]
        chain  = ticker.option_chain(expiry)

        # Current price
        try:
            price = float(ticker.fast_info.last_price)
        except Exception:
            price = float(ticker.history(period="1d")["Close"].iloc[-1])
        if not price or price <= 0:
            return None

        import math

        # Direct column access (same as original fetch_options_pcr that worked)
        call_oi_series = chain.calls["openInterest"].dropna()
        put_oi_series  = chain.puts["openInterest"].dropna()

        # PCR: use min_count=1 so all-NaN → NaN (not 0.0)
        call_oi_total = call_oi_series.sum(min_count=1)
        put_oi_total  = put_oi_series.sum(min_count=1)

        if (math.isnan(float(call_oi_total)) or call_oi_total == 0
                or math.isnan(float(put_oi_total))):
            pcr = None
        else:
            pcr = round(float(put_oi_total) / float(call_oi_total), 3)

        # Build DataFrames for chart (strike + OI only)
        calls_all = chain.calls[["strike", "openInterest"]].rename(columns={"openInterest": "call_oi"})
        puts_all  = chain.puts[["strike", "openInterest"]].rename(columns={"openInterest": "put_oi"})

        # Filter to ±25 % band for a readable chart (needs a valid price)
        if math.isnan(price) or price <= 0:
            return None
        lo, hi = price * 0.75, price * 1.25
        calls = calls_all[(calls_all["strike"] >= lo) & (calls_all["strike"] <= hi)].copy()
        puts  = puts_all[(puts_all["strike"] >= lo) & (puts_all["strike"] <= hi)].copy()

        # Top-3 call walls above price, top-3 put walls below price
        call_above = calls[calls["strike"] > price].nlargest(3, "call_oi")
        put_below  = puts[puts["strike"] < price].nlargest(3, "put_oi")

        # Nearest level from each top-3 group
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
    except Exception:
        return None


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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cik_map() -> dict:
    """Download ticker→CIK from EDGAR (cached 1 hour)."""
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
    """Return list of 8-K/6-K filings within the last `days` days."""
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
                break  # sorted descending, safe to stop
            # parse item numbers, skip nan/empty
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


def analyze(symbol: str):
    df = fetch_history(symbol)
    if df.empty or len(df) < 50:
        return None

    df = df[["Close", "High", "Low", "Volume"]].copy()
    df["MA20"]    = df["Close"].rolling(20).mean()
    df["MA50"]    = df["Close"].rolling(50).mean()
    df["RSI"]     = calc_rsi(df["Close"])
    df["MACD"], df["Signal"] = calc_macd(df["Close"])
    df["VolMA20"] = df["Volume"].rolling(20).mean()

    # KDJ → J值
    df["K"], df["D"], df["J"] = calc_kdj(df)

    # 布林带挤压
    bb_mid         = df["Close"].rolling(20).mean()
    bb_std         = df["Close"].rolling(20).std()
    df["BB_Width"] = 4 * bb_std / bb_mid * 100          # (Upper-Lower)/Mid*100
    df["BB_W_MA60"]= df["BB_Width"].rolling(60, min_periods=30).mean()

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
            "J值":         round(j_val, 1),
            "J状态":       j_flag,
            "BB挤压":      bb_flag,
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

# ── 8-K / 6-K 重大事件 ────────────────────────────────────────────────────────

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
                    "公司":   f["symbol"],
                    "提交日期": f["date"],
                    "文件类型": f["form"],
                    "条目":   item,
                    "说明":   ITEM_DESC.get(item, "其他事项"),
                })
        else:
            rows_8k.append({
                "公司":   f["symbol"],
                "提交日期": f["date"],
                "文件类型": f["form"],
                "条目":   "—",
                "说明":   "（条目信息不适用）",
            })
    st.dataframe(pd.DataFrame(rows_8k), use_container_width=True, hide_index=True)
else:
    st.info("过去10天内，以上股票均无 8-K / 6-K 提交记录。")

st.divider()

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

    # ── indicators table + PCR card ───────────────────────────────────────
    gw = fetch_gamma_wall(sym)

    tbl_col, pcr_col = st.columns([8, 2])
    with tbl_col:
        styled = highlight_signals(scan_df)
        st.dataframe(styled, use_container_width=True, hide_index=True)
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
            msg = "无期权数据" if gw is None else "数据不足"
            st.markdown(
                f"""<div style="border:1px solid #ddd; border-radius:10px;
                    padding:14px; text-align:center; margin-top:6px; color:#aaa;">
                    <div style="font-size:11px;">期权情绪 (PCR)</div>
                    <div style="font-size:13px; margin-top:6px;">{msg}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Gamma 墙柱状图 ────────────────────────────────────────────────────
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
            st.warning(f"图表加载失败（{e}），显示文字版：")

        # 文字总结 — 图表正常或回退时都显示
        st.markdown(
            f"📍 当前价 **${price:.2f}**　│　"
            f"🔴 上方最近阻力 (Call Wall) **{nc}**　│　"
            f"🟢 下方最近支撑 (Put Wall) **{np_}**"
        )
        st.caption(f"Call Walls Top3: {cw}　　　Put Walls Top3: {pw}")

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
