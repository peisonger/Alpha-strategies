"""
pages/2_KR_Market.py
──────────────────────
한국 매크로 지수 대시보드.

포함 데이터:
    - KOSPI · KOSDAQ · KOSPI200
    - K-VIX (공포지수)
    - USD/KRW 환율
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_kr_index_wide
from utils.indicators  import rolling_corr, pct_returns, historical_volatility
from utils.charts      import (
    correlation_heatmap, cumulative_returns_chart,
    monthly_returns_heatmap, apply_dark_theme, DARK_BG, PANEL_BG,
    BORDER, TEXT_COLOR, MUTED, GREEN, RED, BLUE, ORANGE, PURPLE,
)
from utils.performance import monthly_returns_pivot
from utils.export_download import download_csv_button
from utils.indicators import historical_volatility

st.set_page_config(page_title="한국 시장", page_icon="🇰🇷", layout="wide")
st.markdown("# 한국 매크로 시장 대시보드")
st.markdown("---")

ALL_CODES = ["KOSPI", "KOSDAQ", "KOSPI200", "K-VIX", "F-USDKRW"]
COLORS    = {"KOSPI": BLUE, "KOSDAQ": GREEN, "KOSPI200": ORANGE, "K-VIX": RED, "F-USDKRW": PURPLE}

# ─── 사이드바 ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 한국 시장")
    selected = st.multiselect("지수 선택", ALL_CODES,
                               default=["KOSPI", "KOSDAQ", "K-VIX"])
    col_s, col_e = st.columns(2)
    start_date = col_s.date_input("시작일", value=pd.Timestamp("2010-01-01").date())
    end_date   = col_e.date_input("종료일", value=pd.Timestamp("2023-03-01").date())

    roll_window = st.slider("롤링 상관관계 창 (일)", 20, 252, 60)
    st.markdown("---")
    show_norm = st.checkbox("정규화 비교 (기준=100)", value=True)
    show_corr = st.checkbox("상관관계 히트맵", value=True)
    show_vix  = st.checkbox("K-VIX 차트", value=True)
    show_krw  = st.checkbox("USD/KRW 차트", value=True)
    show_month = st.checkbox("월별 수익률 히트맵", value=False)

# ─── 데이터 로드 ──────────────────────────────────────────────────────────────
with st.spinner("한국 지수 데이터 로딩 중…"):
    wide = load_kr_index_wide(
        codes=ALL_CODES,
        start_date=str(start_date),
        end_date=str(end_date),
    )

if wide.empty:
    st.warning("데이터가 없습니다.")
    st.stop()

# 메트릭 계산
def get_metric(code):
    if code not in wide.columns or wide[code].dropna().empty:
        return None, None
    s = wide[code].dropna()
    last = s.iloc[-1]
    chg  = (s.iloc[-1] / s.iloc[-2] - 1) * 100 if len(s) > 1 else 0
    return last, chg

# ─── 상단 메트릭 ──────────────────────────────────────────────────────────────
cols = st.columns(len(ALL_CODES))
for i, code in enumerate(ALL_CODES):
    val, chg = get_metric(code)
    if val is not None:
        label = "VIX" if code == "K-VIX" else ("KRW" if "USD" in code else code)
        fmt   = f"{val:,.2f}" if code == "F-USDKRW" else f"{val:,.2f}"
        cols[i].metric(label, fmt, f"{chg:+.2f}%" if chg is not None else "")

st.markdown("---")

# ─── 정규화 멀티 라인 차트 ────────────────────────────────────────────────────
equity_codes = [c for c in selected if c not in ("K-VIX", "F-USDKRW")]

if equity_codes:
    st.markdown("### 📈 지수 비교 차트")
    fig = go.Figure()
    for code in equity_codes:
        if code not in wide.columns:
            continue
        s = wide[code].dropna()
        y = (s / s.iloc[0] * 100) if show_norm else s
        fig.add_trace(go.Scatter(
            x=s.index, y=y,
            mode="lines", name=code,
            line=dict(color=COLORS.get(code, BLUE), width=2),
            hovertemplate=f"{code}: %{{y:.1f}}<extra></extra>",
        ))
    apply_dark_theme(fig)
    fig.update_layout(
        title="지수 정규화 비교 (기준=100)" if show_norm else "지수 절대값 비교",
        yaxis_title="기준 100" if show_norm else "지수",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── K-VIX ───────────────────────────────────────────────────────────────────
if show_vix and "K-VIX" in wide.columns:
    st.markdown("### 😨 K-VIX 공포지수")
    vix = wide["K-VIX"].dropna()
    fig_vix = go.Figure()
    fig_vix.add_trace(go.Scatter(
        x=vix.index, y=vix,
        mode="lines", fill="tozeroy",
        line=dict(color=RED, width=1.5),
        fillcolor="rgba(248,81,73,0.15)",
        name="K-VIX",
    ))
    fig_vix.add_hline(y=25, line=dict(color=ORANGE, dash="dash"), annotation_text="경계(25)")
    fig_vix.add_hline(y=35, line=dict(color=RED,    dash="dash"), annotation_text="공포(35)")
    apply_dark_theme(fig_vix)
    fig_vix.update_layout(title="K-VIX (한국 변동성지수)", height=300)
    st.plotly_chart(fig_vix, use_container_width=True)

# ─── USD/KRW ─────────────────────────────────────────────────────────────────
if show_krw and "F-USDKRW" in wide.columns:
    st.markdown("### 💱 USD/KRW 환율")
    krw = wide["F-USDKRW"].dropna()
    fig_krw = go.Figure()
    fig_krw.add_trace(go.Scatter(
        x=krw.index, y=krw,
        mode="lines",
        line=dict(color=PURPLE, width=1.5),
        name="USD/KRW",
    ))
    apply_dark_theme(fig_krw)
    fig_krw.update_layout(title="USD/KRW 환율", yaxis_title="원", height=300)
    st.plotly_chart(fig_krw, use_container_width=True)

# ─── 상관관계 히트맵 ──────────────────────────────────────────────────────────
if show_corr and len(selected) >= 2:
    st.markdown("### 🔗 상관관계 분석")
    avail = [c for c in selected if c in wide.columns]
    ret_df = wide[avail].pct_change().dropna()
    corr = ret_df.corr()

    c1, c2 = st.columns(2)
    with c1:
        fig_corr = correlation_heatmap(corr, title=f"수익률 상관관계 (롤링 {roll_window}일)")
        st.plotly_chart(fig_corr, use_container_width=True)
    with c2:
        # 롤링 상관관계 (첫 두 지수 간)
        if len(avail) >= 2:
            s1 = ret_df[avail[0]]
            s2 = ret_df[avail[1]]
            roll = rolling_corr(s1, s2, window=roll_window)
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(
                x=roll.index, y=roll,
                mode="lines", line=dict(color=BLUE, width=1.5),
                name=f"{avail[0]} vs {avail[1]}",
            ))
            fig_roll.add_hline(y=0, line=dict(color=MUTED, dash="dash"))
            apply_dark_theme(fig_roll)
            fig_roll.update_layout(
                title=f"롤링 상관관계: {avail[0]} vs {avail[1]} ({roll_window}일)",
                yaxis=dict(range=[-1, 1]),
                height=400,
            )
            st.plotly_chart(fig_roll, use_container_width=True)

# ─── 월별 수익률 히트맵 ───────────────────────────────────────────────────────
if show_month:
    st.markdown("### 📅 월별 수익률")
    base_code = equity_codes[0] if equity_codes else None
    if base_code and base_code in wide.columns:
        ret = wide[base_code].dropna().pct_change().dropna()
        pivot = monthly_returns_pivot(ret)
        fig_month = monthly_returns_heatmap(pivot, title=f"{base_code} 월별 수익률")
        st.plotly_chart(fig_month, use_container_width=True)

# ─── 실현변동성 (선택 지수) ───────────────────────────────────────────────────
equity_for_vol = [c for c in selected if c in wide.columns and c not in ("K-VIX", "F-USDKRW")]
if equity_for_vol:
    with st.expander("선택 지수 연환산 실현변동성 (60거래일)", expanded=False):
        fig_v = go.Figure()
        for code in equity_for_vol:
            s = wide[code].dropna()
            vol60 = historical_volatility(s, window=60, annualize=True)
            fig_v.add_trace(go.Scatter(
                x=vol60.index, y=vol60 * 100,
                mode="lines", name=code,
                line=dict(color=COLORS.get(code, BLUE), width=1.5),
            ))
        apply_dark_theme(fig_v)
        fig_v.update_layout(
            title="60일 롤링 실현변동성 (연율, %)",
            yaxis_title="변동성 (%)",
            height=360,
        )
        st.plotly_chart(fig_v, use_container_width=True)

# ─── CSV 내보내기 ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 데이터 내보내기")
with st.expander("현재 구간 지수 시계열 CSV", expanded=False):
    export_wide = wide.reset_index()
    if export_wide.columns[0] != "date":
        export_wide = export_wide.rename(columns={export_wide.columns[0]: "date"})
    download_csv_button(
        export_wide,
        file_name=f"kr_index_wide_{start_date}_{end_date}.csv",
        label="지수 데이터 다운로드",
        key="dl_kr_wide",
    )
