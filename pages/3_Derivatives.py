"""
pages/3_Derivatives.py
───────────────────────
한국 파생상품 대시보드.

포함:
    - K200F / K200MF / KQ150F 선물 비교
    - KOSPI200 옵션 체인 (행사가별 OI 히트맵)
    - Put/Call Ratio
    - IV 스마일 (내재변동성 근사)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
from functools import reduce
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.export_download import download_csv_button

from utils.data_loader import load_all_futures, load_options
from utils.indicators  import pct_returns, historical_volatility, rolling_corr
from utils.charts      import (
    apply_dark_theme, correlation_heatmap, cumulative_returns_chart,
    BLUE, GREEN, RED, ORANGE, PURPLE, PANEL_BG, BORDER, TEXT_COLOR, MUTED,
)

st.set_page_config(page_title="파생상품", page_icon="📉", layout="wide")
st.markdown("# 한국 파생상품 대시보드")
st.markdown("---")

FUT_COLORS = {"K200F": BLUE, "K200MF": GREEN, "KQ150F": ORANGE}

# ─── 사이드바 ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 파생상품")
    col_s, col_e = st.columns(2)
    start_date = col_s.date_input("시작일", pd.Timestamp("2010-01-01").date())
    end_date   = col_e.date_input("종료일", pd.Timestamp("2024-02-01").date())

    st.markdown("### 옵션 설정")
    opt_ym = st.selectbox("옵션 만기 월", ["201903","201906","201909","201912",
                                           "202001","202006","202012",
                                           "202101","202106","202112"], index=3)
    show_spread   = st.checkbox("선물 스프레드 분석", value=True)
    show_chain    = st.checkbox("옵션 체인 히트맵",   value=True)
    show_pcr      = st.checkbox("풋/콜 비율", value=True)
    show_iv_smile = st.checkbox("IV 스마일 (HV 근사)", value=True)

# ─── 선물 데이터 로드 ─────────────────────────────────────────────────────────
with st.spinner("선물 데이터 로딩…"):
    futures = load_all_futures(start_date=str(start_date), end_date=str(end_date))

# ─── 상단 메트릭 ──────────────────────────────────────────────────────────────
m_cols = st.columns(3)
for i, (sym, df_f) in enumerate(futures.items()):
    if df_f.empty:
        continue
    last = df_f["close"].iloc[-1]
    chg  = (df_f["close"].iloc[-1] / df_f["close"].iloc[-2] - 1) * 100
    m_cols[i].metric(sym, f"{last:,.2f}", f"{chg:+.2f}%")

st.markdown("---")

# ─── 선물 비교 차트 (정규화) ──────────────────────────────────────────────────
st.markdown("### 📊 선물 비교 차트")
fig_fut = go.Figure()
for sym, df_f in futures.items():
    if df_f.empty:
        continue
    norm = df_f["close"] / df_f["close"].iloc[0] * 100
    fig_fut.add_trace(go.Scatter(
        x=df_f["date"], y=norm,
        mode="lines", name=sym,
        line=dict(color=FUT_COLORS[sym], width=1.8),
        hovertemplate=f"{sym}: %{{y:.1f}}<extra></extra>",
    ))
apply_dark_theme(fig_fut)
fig_fut.update_layout(title="한국 선물 정규화 비교 (기준=100)", yaxis_title="기준 100", height=400)
st.plotly_chart(fig_fut, use_container_width=True)

# ─── 스프레드 분석 ────────────────────────────────────────────────────────────
if show_spread:
    st.markdown("### 📐 K200F vs K200MF 스프레드")
    df_k200  = futures["K200F"]
    df_k200m = futures["K200MF"]

    if not df_k200.empty and not df_k200m.empty:
        merged = pd.merge(
            df_k200[["date", "close"]].rename(columns={"close": "K200F"}),
            df_k200m[["date", "close"]].rename(columns={"close": "K200MF"}),
            on="date", how="inner",
        )
        merged["spread"]   = merged["K200F"] - merged["K200MF"]
        merged["spread_pct"] = merged["spread"] / merged["K200F"] * 100

        fig_sp = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.6, 0.4], vertical_spacing=0.03)
        for col, color in [("K200F", BLUE), ("K200MF", GREEN)]:
            fig_sp.add_trace(go.Scatter(
                x=merged["date"], y=merged[col],
                mode="lines", name=col,
                line=dict(color=color, width=1.5),
            ), row=1, col=1)

        fig_sp.add_trace(go.Bar(
            x=merged["date"], y=merged["spread_pct"],
            marker_color=[GREEN if v >= 0 else RED for v in merged["spread_pct"]],
            name="스프레드 (%)", opacity=0.8,
        ), row=2, col=1)

        apply_dark_theme(fig_sp)
        fig_sp.update_layout(title="K200F - K200MF 스프레드", height=450)
        st.plotly_chart(fig_sp, use_container_width=True)

# ─── 옵션 데이터 로드 ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### 🎯 KOSPI200 옵션 분석 — 만기: {opt_ym}")

with st.spinner(f"옵션 데이터 로딩 중… ({opt_ym})"):
    opt_df = load_options(year_month=opt_ym)

if opt_df.empty:
    st.warning(f"{opt_ym} 만기 옵션 데이터가 없습니다.")
else:
    # ─── 옵션 체인 히트맵 ─────────────────────────────────────────────────────
    if show_chain:
        st.markdown("#### 옵션 체인 — 행사가별 거래대금")

        # 마지막 거래일 데이터만 사용
        last_date = opt_df["dateint"].max()
        chain_df  = opt_df[opt_df["dateint"] == last_date]

        calls = chain_df[chain_df["odir"] ==  1].groupby("f_prc")["trd_val"].sum().rename("콜")
        puts  = chain_df[chain_df["odir"] == -1].groupby("f_prc")["trd_val"].sum().rename("풋")
        chain = pd.concat([calls, puts], axis=1).fillna(0).reset_index()
        chain = chain.sort_values("f_prc")

        fig_chain = go.Figure()
        fig_chain.add_trace(go.Bar(
            x=chain["f_prc"], y=chain["콜"] / 1e9,
            name="콜 (거래대금)", marker_color=GREEN, opacity=0.8,
        ))
        fig_chain.add_trace(go.Bar(
            x=chain["f_prc"], y=-(chain["풋"] / 1e9),
            name="풋 (거래대금)", marker_color=RED, opacity=0.8,
        ))
        apply_dark_theme(fig_chain)
        fig_chain.update_layout(
            title=f"옵션 체인 — {last_date} (단위: 십억원)",
            barmode="overlay",
            xaxis_title="행사가",
            yaxis_title="거래대금 (십억원)",
            height=400,
        )
        st.plotly_chart(fig_chain, use_container_width=True)

    # ─── Put/Call Ratio ────────────────────────────────────────────────────────
    if show_pcr:
        st.markdown("#### 풋/콜 비율 (거래대금 기준)")
        daily = opt_df.groupby(["dateint", "odir"])["trd_val"].sum().unstack(fill_value=0)
        daily.columns = ["풋" if c == -1 else "콜" for c in daily.columns]

        if "풋" in daily.columns and "콜" in daily.columns:
            pcr = daily["풋"] / daily["콜"].replace(0, np.nan)
            dates = pd.to_datetime(daily.index.astype(str), format="%Y%m%d")

            fig_pcr = go.Figure()
            fig_pcr.add_trace(go.Scatter(
                x=dates, y=pcr,
                mode="lines", fill="tozeroy",
                line=dict(color=ORANGE, width=1.5),
                fillcolor="rgba(210,153,34,0.15)",
                name="풋/콜 비율",
            ))
            fig_pcr.add_hline(y=1.0, line=dict(color=MUTED, dash="dash"),
                              annotation_text="균형(1.0)")
            apply_dark_theme(fig_pcr)
            fig_pcr.update_layout(
                title="풋/콜 비율 (>1: 풋 우세 = 하락 베팅↑)",
                height=300,
            )
            st.plotly_chart(fig_pcr, use_container_width=True)

    # ─── IV 스마일 ─────────────────────────────────────────────────────────────
    if show_iv_smile:
        st.markdown("#### 📈 IV 스마일 (HV 근사)")

        # HV 근사: 행사가별 수익률 변동성 사용
        last_date = opt_df["dateint"].max()
        smile_df  = opt_df[opt_df["dateint"] == last_date].copy()
        smile_df["mid"] = (smile_df["open__l1d"] + smile_df["close"]) / 2
        smile_df["approx_iv"] = (smile_df["close"] / smile_df["f_prc"]) * 100  # 근사

        call_smile = smile_df[smile_df["odir"] ==  1].groupby("f_prc")["approx_iv"].mean()
        put_smile  = smile_df[smile_df["odir"] == -1].groupby("f_prc")["approx_iv"].mean()

        fig_smile = go.Figure()
        fig_smile.add_trace(go.Scatter(
            x=call_smile.index, y=call_smile,
            mode="lines+markers", name="콜 IV (근사)",
            line=dict(color=GREEN, width=2),
        ))
        fig_smile.add_trace(go.Scatter(
            x=put_smile.index, y=put_smile,
            mode="lines+markers", name="풋 IV (근사)",
            line=dict(color=RED, width=2),
        ))
        apply_dark_theme(fig_smile)
        fig_smile.update_layout(
            title=f"IV 스마일 — {last_date} (프리미엄/행사가 근사)",
            xaxis_title="행사가",
            yaxis_title="IV 근사값 (%)",
            height=350,
        )
        st.plotly_chart(fig_smile, use_container_width=True)
        st.caption("⚠️ 실제 IV는 Black-Scholes 역산이 필요합니다. 여기서는 프리미엄/행사가 비율로 근사합니다.")

# ─── 데이터 내보내기 ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 데이터 내보내기")

with st.expander("선물 일별 종가 (K200F · K200MF · KQ150F)", expanded=False):
    fut_parts = []
    for sym, df_f in futures.items():
        if df_f is None or df_f.empty:
            continue
        fut_parts.append(df_f[["date", "close"]].rename(columns={"close": sym}))
    if fut_parts:
        fut_merged = reduce(
            lambda a, b: pd.merge(a, b, on="date", how="outer"),
            fut_parts,
        ).sort_values("date")
        download_csv_button(
            fut_merged,
            file_name=f"futures_close_{start_date}_{end_date}.csv",
            label="선물 종가 CSV 다운로드",
            key="dl_futures",
        )
    else:
        st.caption("선물 데이터가 없습니다.")

with st.expander("옵션 데이터 CSV (행 제한)", expanded=False):
    if opt_df.empty:
        st.caption("옵션 데이터가 없습니다.")
    else:
        max_rows = 100_000
        opt_out = opt_df if len(opt_df) <= max_rows else opt_df.iloc[:max_rows].copy()
        st.caption(f"최대 {max_rows:,}행까지 저장 (전체 {len(opt_df):,}행).")
        download_csv_button(
            opt_out,
            file_name=f"kospi200_option_{opt_ym}.csv",
            label="옵션 데이터 다운로드",
            key="dl_opt",
        )
