"""
pages/1_BTC_Trading.py
───────────────────────
BTC/USDT 멀티타임프레임 트레이딩 대시보드.

기능:
    - 타임프레임 전환 (5m / 1h / 6h / 1d / 1w)
    - 캔들스틱 차트 + 오버레이 지표
    - RSI · MACD 서브플롯
    - Volume Profile
    - 성과 요약 메트릭
"""

import streamlit as st
import pandas as pd

# utils 경로 추가
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader  import load_btc, benchmark
from utils.indicators   import add_all_indicators, bollinger_bands, volume_profile, macd, rsi, drawdown
from utils.backtest_simple import ma_crossover_returns
from utils.export_download import download_csv_button
from utils.performance  import summary_stats, format_stats_df
from utils.charts       import (
    candlestick_chart, rsi_chart, macd_chart,
    volume_profile_chart, cumulative_returns_chart, apply_dark_theme,
)
import plotly.graph_objects as go

# ─── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="BTC 거래", page_icon="₿", layout="wide")
st.markdown("# BTC/USDT 멀티 타임프레임 대시보드")
st.markdown("---")

# ─── 사이드바 컨트롤 ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## BTC 설정")

    tf = st.selectbox(
        "타임프레임",
        options=["1d", "6h", "1h", "5m", "1w"],
        index=0,
        help="5m 데이터는 약 20만 행으로 로딩에 다소 시간이 걸릴 수 있습니다.",
    )

    st.markdown("### 날짜 범위")
    col_s, col_e = st.columns(2)
    start_date = col_s.date_input("시작일", value=pd.Timestamp("2021-01-01").date())
    end_date   = col_e.date_input("종료일", value=pd.Timestamp("2025-01-01").date())

    st.markdown("### 지표 선택")
    show_sma20  = st.checkbox("이평 20", value=True)
    show_sma50  = st.checkbox("이평 50", value=True)
    show_sma200 = st.checkbox("이평 200", value=False)
    show_bb     = st.checkbox("볼린저 밴드", value=True)
    show_volume = st.checkbox("거래량", value=True)
    show_rsi    = st.checkbox("RSI", value=True)
    show_macd   = st.checkbox("MACD", value=True)
    show_vp     = st.checkbox("거래량 프로파일", value=False)

    st.markdown("---")
    if st.button("캐시 초기화", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ─── 데이터 로드 ──────────────────────────────────────────────────────────────
with st.spinner(f"BTC {tf} 데이터 로딩 중…"):
    df, load_ms = benchmark(
        load_btc, tf=tf,
        start_date=str(start_date),
        end_date=str(end_date),
        label=f"BTC {tf} 로드",
    )

if df.empty:
    st.warning("선택한 날짜 범위에 데이터가 없습니다.")
    st.stop()

# 지표 계산
df = add_all_indicators(df)

# ─── 상단 메트릭 ──────────────────────────────────────────────────────────────
latest  = df["Close"].iloc[-1]
prev    = df["Close"].iloc[-2]
change  = (latest - prev) / prev * 100
high52  = df["Close"].tail(365).max() if tf == "1d" else df["Close"].max()
low52   = df["Close"].tail(365).min() if tf == "1d" else df["Close"].min()
rsi_val = df["rsi_14"].iloc[-1]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("BTC 현재가",  f"${latest:,.0f}", f"{change:+.2f}%")
m2.metric("52주 고점",   f"${high52:,.0f}")
m3.metric("52주 저점",   f"${low52:,.0f}")
m4.metric("RSI (14)",   f"{rsi_val:.1f}",
          "과매수 ⚠️" if rsi_val > 70 else ("과매도 🟢" if rsi_val < 30 else "중립"))
m5.metric("데이터 행수", f"{len(df):,}행", f"{load_ms:.0f}ms 로딩")

st.markdown("---")

# ─── 캔들스틱 차트 ────────────────────────────────────────────────────────────
indicators_overlay = []
if show_sma20:
    indicators_overlay.append({"name": "이평 20",  "data": df["sma_20"],  "color": "#58a6ff"})
if show_sma50:
    indicators_overlay.append({"name": "이평 50",  "data": df["sma_50"],  "color": "#d29922"})
if show_sma200:
    indicators_overlay.append({"name": "이평 200", "data": df["sma_200"], "color": "#bc8cff"})
if show_bb:
    indicators_overlay.extend([
        {"name": "볼린저 상단", "data": df["bb_upper"], "color": "#8b949e"},
        {"name": "볼린저 중간", "data": df["bb_mid"],   "color": "#8b949e"},
        {"name": "볼린저 하단", "data": df["bb_lower"], "color": "#8b949e"},
    ])

fig_candle = candlestick_chart(
    df=df,
    date_col="datetime",
    title=f"BTC/USDT {tf.upper()} — {str(start_date)} ~ {str(end_date)}",
    show_volume=show_volume,
    indicators=indicators_overlay,
)
st.plotly_chart(fig_candle, use_container_width=True)

# ─── RSI / MACD 서브플롯 ──────────────────────────────────────────────────────
col_rsi, col_macd = st.columns(2)

if show_rsi:
    with col_rsi:
        fig_rsi = rsi_chart(df["rsi_14"], df["datetime"])
        st.plotly_chart(fig_rsi, use_container_width=True)

if show_macd:
    with col_macd:
        macd_cols = df[["macd_line", "signal_line", "histogram"]]
        fig_macd = macd_chart(macd_cols, df["datetime"])
        st.plotly_chart(fig_macd, use_container_width=True)

# ─── Volume Profile ───────────────────────────────────────────────────────────
if show_vp:
    st.markdown("### 거래량 프로파일")
    vp_df = volume_profile(df, bins=40)
    fig_vp = volume_profile_chart(vp_df, title=f"BTC 거래량 프로파일 ({tf})")
    st.plotly_chart(fig_vp, use_container_width=True)

# ─── 성과 요약 ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 성과 요약")

col_perf, col_ret = st.columns([1, 2])
with col_perf:
    stats = summary_stats(df["pct_ret"], freq=365 if tf in ["1d","1w"] else 365*24, label="BTC 매수·보유")
    stats_df = format_stats_df(stats)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

with col_ret:
    cum_dict = {"BTC 매수·보유": df["pct_ret"].set_axis(df["datetime"])}
    fig_cum = cumulative_returns_chart(cum_dict, title="BTC 누적 수익률")
    st.plotly_chart(fig_cum, use_container_width=True)

# ─── 낙폭(Underwater) 차트 ───────────────────────────────────────────────────
st.markdown("### 누적 손익 기준 낙폭")
eq_curve = (1 + df["pct_ret"].fillna(0)).cumprod()
uw_pct = drawdown(eq_curve) * 100
fig_uw = go.Figure(go.Scatter(
    x=df["datetime"], y=uw_pct, mode="lines",
    fill="tozeroy",
    line=dict(color="#f85149", width=1),
    fillcolor="rgba(248,81,73,0.25)",
    name="낙폭 %",
))
apply_dark_theme(fig_uw)
fig_uw.update_layout(
    title="매수·보유 가정 시 최고점 대비 하락폭 (%)",
    yaxis_title="낙폭 (%)",
    height=280,
    showlegend=False,
)
st.plotly_chart(fig_uw, use_container_width=True)

# ─── 간단 백테스트 ───────────────────────────────────────────────────────────
with st.expander("간단 백테스트: 이평 크로스 vs 매수·보유", expanded=False):
    st.caption("빠른 이평 > 느린 이평일 때만 롱. 슬리피지·수수료·레버리지 미반영.")
    c_bs1, c_bs2 = st.columns(2)
    fast_w = c_bs1.slider("빠른 이평", 5, 55, 20, key="bt_fast")
    slow_w = c_bs2.slider("느린 이평", 15, 250, 50, key="bt_slow")
    if fast_w >= slow_w:
        st.warning("빠른 이평 기간은 느린 이평보다 작아야 합니다.")
    else:
        strat_ret = ma_crossover_returns(df["Close"], fast_w, slow_w)
        strat_cum = (1 + strat_ret.fillna(0)).cumprod() - 1
        bh_cum = (1 + df["pct_ret"].fillna(0)).cumprod() - 1
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=df["datetime"], y=strat_cum * 100,
            mode="lines", name=f"이평({fast_w}/{slow_w}) 전략",
            line=dict(color="#58a6ff", width=1.8),
        ))
        fig_bt.add_trace(go.Scatter(
            x=df["datetime"], y=bh_cum * 100,
            mode="lines", name="매수·보유",
            line=dict(color="#8b949e", width=1.2),
        ))
        apply_dark_theme(fig_bt)
        fig_bt.update_layout(
            title="누적 수익률 비교 (%)",
            yaxis_title="누적 (%)",
            height=380,
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_bt, use_container_width=True)
        m1, m2 = st.columns(2)
        m1.metric(
            "전략 누적 수익",
            f"{strat_cum.iloc[-1]:.2%}",
        )
        m2.metric(
            "매수·보유 누적 수익",
            f"{bh_cum.iloc[-1]:.2%}",
        )

# ─── 데이터 내보내기 ─────────────────────────────────────────────────────────
with st.expander("OHLCV 데이터 CSV 저장", expanded=False):
    export_df = df[
        ["datetime", "Open", "High", "Low", "Close", "Volume", "pct_ret"]
    ].copy()
    export_df = export_df.rename(columns={"pct_ret": "수익률"})
    download_csv_button(
        export_df,
        file_name=f"btc_{tf}_{start_date}_{end_date}.csv",
        label="현재 구간 OHLCV + 일간수익률 다운로드",
        key="dl_btc_ohlcv",
    )
