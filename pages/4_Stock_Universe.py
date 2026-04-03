"""
pages/4_Stock_Universe.py
──────────────────────────
전종목 주가 스크리너 (adjchart 780만 행).


기능:
    - 종목코드 / 날짜 범위 고속 필터링
    - 외국인·기관 수급 분석
    - 시가총액 상위 종목 스크리너
    - 수익률 팩터 정렬
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_stock_universe
from utils.kr_stock_names import stock_select_label, stock_short_label
from utils.export_download import download_csv_button
from utils.indicators  import pct_returns, sma, rsi, historical_volatility
from utils.charts      import (
    candlestick_chart, apply_dark_theme,
    BLUE, GREEN, RED, ORANGE, PURPLE, MUTED,
)

_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT / "data"
_adj_candidates = sorted(_DATA_DIR.glob("adjchart_df*.parquet"))
_DEFAULT_ADJCHART = str(_adj_candidates[0]) if _adj_candidates else ""

# 시총·거래대금 상위 위주 대표 종목 (드롭다운 기본으로 넉넉히 보기)
_DEFAULT_CODES_STR = (
    "A005930,A000660,A035420,A051910,A006400,A035720,A207940,A068270,A105560,A055550,"
    "A096770,A015760,A066570,A017670,A028260,A034730,A003550,A009150,A010130,A018260,"
    "A032830,A316140,A024110,A086280,A090430,A251270,A128940,A302440,A323410,A011200,"
    "A030200,A036570,A033780,A010950,A137310"
)

st.set_page_config(page_title="전종목 스크리너", page_icon="🔍", layout="wide")
st.markdown("# 전종목 스크리너")
st.markdown("---")

# ─── 사이드바 ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 전종목 스크리너")

    parquet_path = st.text_input(
        "수정주가 Parquet 경로",
        value=_DEFAULT_ADJCHART,
        placeholder="예: …/data/adjchart_df.20100101-20231013.parquet",
        help="비우면 data/adjchart_df*.parquet 가 있으면 자동 기본값이 채워집니다.",
    )

    load_all_stocks = st.checkbox(
        "이 기간 전 종목 로드",
        value=False,
        help="체크 시 종목 입력은 무시됩니다. 행 수가 매우 많으면 수 분·대용량 메모리가 필요할 수 있습니다. "
        "먼저 시작·종료일을 좁혀 보세요.",
    )
    codes_input = st.text_input(
        "종목코드 (쉼표 구분)",
        value=_DEFAULT_CODES_STR,
        disabled=load_all_stocks,
        help="체크 해제 시: 적은 종목만 읽어 빠릅니다. 목록을 지우면 해당 기간 전 종목과 동일하게 동작합니다.",
    )
    codes = [c.strip() for c in codes_input.split(",") if c.strip()]
    codes_filter = None if load_all_stocks else (codes if codes else None)

    col_s, col_e = st.columns(2)
    start_di = int(col_s.date_input("시작일", pd.Timestamp("2020-01-01").date())
                .strftime("%Y%m%d"))
    end_di   = int(col_e.date_input("종료일", pd.Timestamp("2023-10-13").date())
                .strftime("%Y%m%d"))

    st.markdown("### 분석 항목")
    show_price   = st.checkbox("주가 차트",        value=True)
    show_supply  = st.checkbox("외국인·기관 수급", value=True)
    show_mc      = st.checkbox("시가총액 추이",    value=True)
    show_screen  = st.checkbox("멀티팩터 스크리너", value=True)

# ─── 샘플 데이터 생성 함수 (Parquet 없을 때) ─────────────────────────────────
def make_sample_data(codes: list[str], start_di: int, end_di: int) -> pd.DataFrame:
    """데모용 샘플 데이터 생성."""
    dates = pd.date_range(
        pd.to_datetime(str(start_di), format="%Y%m%d"),
        pd.to_datetime(str(end_di), format="%Y%m%d"),
        freq="B",
    )
    rows = []
    np.random.seed(42)
    base_prices = {"A005930": 70000, "A000660": 130000, "A035420": 350000}
    for code in codes:
        base = base_prices.get(code, 50000)
        rets = np.random.normal(0.0003, 0.018, len(dates))
        closes = base * np.exp(np.cumsum(rets))
        for i, d in enumerate(dates):
            rows.append({
                "dateint":       int(d.strftime("%Y%m%d")),
                "sh7code":       code,
                "open":          closes[i] * np.random.uniform(0.99, 1.01),
                "high":          closes[i] * np.random.uniform(1.00, 1.02),
                "low":           closes[i] * np.random.uniform(0.98, 1.00),
                "close":         closes[i],
                "vol":           int(np.random.randint(500_000, 5_000_000)),
                "trd_val":       int(closes[i] * np.random.randint(500_000, 5_000_000)),
                "mc":            int(closes[i] * np.random.randint(5_000_000_000, 6_000_000_000)),
                "org_netvol":    int(np.random.randint(-500_000, 500_000)),
                "frg_belong_sh": int(np.random.randint(1_000_000, 10_000_000)),
            })
    return pd.DataFrame(rows)

# ─── 데이터 로드 ──────────────────────────────────────────────────────────────
use_sample = False
df_raw = pd.DataFrame()

if parquet_path and Path(parquet_path).exists():
    with st.spinner("Parquet 고속 로딩 중… (필요한 열만 읽기)"):
        import time
        t0 = time.perf_counter()
        df_raw = load_stock_universe(
            parquet_path=parquet_path,
            codes=codes_filter,
            start_dateint=start_di,
            end_dateint=end_di,
        )
        elapsed = (time.perf_counter() - t0) * 1000
    st.success(f"{len(df_raw):,}행 로드 완료 — **{elapsed:.0f}ms** *(캐시)*")
else:
    use_sample = True
    st.info("저장된 Parquet 파일 경로가 없어 **샘플 데이터**로 화면을 보여 줍니다.\n\n"
            "실제 데이터: 사이드바에 수정주가 Parquet 파일 경로를 입력하세요.")
    demo_codes = (
        [c.strip() for c in _DEFAULT_CODES_STR.split(",") if c.strip()]
        if load_all_stocks or not codes
        else codes
    )
    df_raw = make_sample_data(demo_codes, start_di, end_di)

if df_raw.empty:
    st.warning("데이터가 없습니다. 종목코드 또는 날짜 범위를 확인하세요.")
    st.stop()

df_raw["date"] = pd.to_datetime(df_raw["dateint"].astype(str), format="%Y%m%d")

with st.expander("조회 데이터 CSV 저장", expanded=False):
    dl = df_raw.copy()
    download_csv_button(
        dl,
        file_name=f"stock_universe_{start_di}_{end_di}.csv",
        label=f"전체 조회 결과 다운로드 ({len(dl):,}행)",
        key="dl_stock_raw",
    )
loaded_codes = sorted(df_raw["sh7code"].astype(str).unique().tolist())
loaded_codes_ui = sorted(loaded_codes, key=lambda c: stock_select_label(c).casefold())

# ─── 탭 레이아웃 ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 주가 차트", "🏦 외국인·기관 수급", "💰 시가총액", "🔬 팩터 스크리너"])

# ═══ TAB 1: 주가 차트 ════════════════════════════════════════════════════════
with tab1:
    if show_price:
        selected_code = st.selectbox(
            "종목 선택",
            loaded_codes_ui,
            key="price_code",
            format_func=stock_select_label,
        )
        df_one = df_raw[df_raw["sh7code"] == selected_code].sort_values("date").reset_index(drop=True)

        if df_one.empty:
            st.warning("해당 종목 데이터 없음")
        else:
            # 컬럼명 대문자로 변환 (charts 모듈 호환)
            df_chart = df_one.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "vol": "Volume",
            })

            # SMA 오버레이
            df_chart["sma_20"]  = sma(df_chart["Close"], 20)
            df_chart["sma_60"]  = sma(df_chart["Close"], 60)
            df_chart["sma_120"] = sma(df_chart["Close"], 120)

            indicators = [
                {"name": "SMA 20",  "data": df_chart["sma_20"],  "color": BLUE},
                {"name": "SMA 60",  "data": df_chart["sma_60"],  "color": ORANGE},
                {"name": "SMA 120", "data": df_chart["sma_120"], "color": PURPLE},
            ]

            fig = candlestick_chart(
                df=df_chart, date_col="date",
                title=f"{stock_short_label(selected_code)} 주가",
                show_volume=True, indicators=indicators,
            )
            st.plotly_chart(fig, use_container_width=True)

            # RSI
            df_chart["rsi"] = rsi(df_chart["Close"], 14)
            col_rsi, col_stat = st.columns([2, 1])
            with col_rsi:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=df_chart["date"], y=df_chart["rsi"],
                    mode="lines", line=dict(color=PURPLE, width=1.5), name="RSI 14",
                ))
                fig_rsi.add_hline(y=70, line=dict(color=RED, dash="dash"))
                fig_rsi.add_hline(y=30, line=dict(color=GREEN, dash="dash"))
                apply_dark_theme(fig_rsi)
                fig_rsi.update_layout(title="RSI (14)", height=200,
                                    yaxis=dict(range=[0,100]))
                st.plotly_chart(fig_rsi, use_container_width=True)

            with col_stat:
                last_close = df_chart["Close"].iloc[-1]
                ret_1m     = df_chart["Close"].pct_change(21).iloc[-1]
                ret_3m     = df_chart["Close"].pct_change(63).iloc[-1]
                st.metric("현재가",  f"{last_close:,.0f}원")
                st.metric("1개월 수익률", f"{ret_1m:.1%}")
                st.metric("3개월 수익률", f"{ret_3m:.1%}")

# ═══ TAB 2: 외국인·기관 수급 ═════════════════════════════════════════════════
with tab2:
    if show_supply:
        selected_code2 = st.selectbox(
            "종목 선택",
            loaded_codes_ui,
            key="supply_code",
            format_func=stock_select_label,
        )
        df_sup = df_raw[df_raw["sh7code"] == selected_code2].sort_values("date").reset_index(drop=True)

        if df_sup.empty:
            st.warning("해당 종목 데이터 없음")
        else:
            fig_sup = go.Figure()

            # 주가 (2축)
            fig_sup.add_trace(go.Scatter(
                x=df_sup["date"], y=df_sup["close"],
                mode="lines", name="주가",
                line=dict(color=BLUE, width=1.5),
                yaxis="y2",
            ))

            # 기관 순매수 누적
            if "org_netvol" in df_sup.columns:
                org_cum = df_sup["org_netvol"].cumsum()
                fig_sup.add_trace(go.Bar(
                    x=df_sup["date"], y=org_cum,
                    name="기관 누적 순매수",
                    marker_color=[GREEN if v >= 0 else RED for v in org_cum],
                    opacity=0.7,
                ))

            apply_dark_theme(fig_sup)
            fig_sup.update_layout(
                title=f"{stock_short_label(selected_code2)} — 기관 수급 vs 주가",
                yaxis=dict(title="기관 누적 순매수 (주)"),
                yaxis2=dict(title="주가 (원)", overlaying="y", side="right",
                            showgrid=False),
                barmode="relative",
                height=400,
                legend=dict(orientation="h", y=1.05),
            )
            st.plotly_chart(fig_sup, use_container_width=True)

            # 외국인 보유 비중
            if "frg_belong_sh" in df_sup.columns and "mc" in df_sup.columns:
                frg_ratio = df_sup["frg_belong_sh"] / (df_sup["mc"] / df_sup["close"].replace(0, np.nan)) * 100
                fig_frg = go.Figure()
                fig_frg.add_trace(go.Scatter(
                    x=df_sup["date"], y=frg_ratio,
                    mode="lines", fill="tozeroy",
                    line=dict(color=ORANGE, width=1.5),
                    fillcolor="rgba(210,153,34,0.15)",
                    name="외국인 보유 비중",
                ))
                apply_dark_theme(fig_frg)
                fig_frg.update_layout(
                    title=f"{stock_short_label(selected_code2)} — 외국인 보유 비중 (%)",
                    yaxis_title="%", height=250,
                )
                st.plotly_chart(fig_frg, use_container_width=True)

# ═══ TAB 3: 시가총액 ══════════════════════════════════════════════════════════
with tab3:
    if show_mc:
        st.markdown("### 💰 종목별 시가총액 비교")

        # 날짜별 시가총액 (단위: 조원)
        mc_df = df_raw.groupby(["date", "sh7code"])["mc"].last().unstack(fill_value=np.nan)
        mc_df = mc_df / 1e12  # 조원

        fig_mc = go.Figure()
        colors = [BLUE, GREEN, ORANGE, PURPLE, RED]
        for i, code in enumerate(mc_df.columns):
            c = str(code)
            lbl = stock_select_label(c)
            fig_mc.add_trace(go.Scatter(
                x=mc_df.index, y=mc_df[code],
                mode="lines", name=lbl,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f"{lbl}: %{{y:.1f}}조원<extra></extra>",
            ))
        apply_dark_theme(fig_mc)
        fig_mc.update_layout(
            title="시가총액 추이 (조원)", yaxis_title="조원", height=400,
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        # 최신 시가총액 바 차트
        latest_date = df_raw["date"].max()
        mc_latest = (
            df_raw[df_raw["date"] == latest_date]
            .set_index("sh7code")["mc"]
            .sort_values(ascending=False) / 1e12
        )
        bar_x = [stock_select_label(str(c)) for c in mc_latest.index]
        fig_bar = go.Figure(go.Bar(
            x=bar_x, y=mc_latest.values,
            marker_color=[colors[i % len(colors)] for i in range(len(mc_latest))],
            text=[f"{v:.1f}조" for v in mc_latest.values],
            textposition="outside",
        ))
        apply_dark_theme(fig_bar)
        fig_bar.update_layout(title=f"최신 시가총액 ({latest_date.date()})", height=300)
        st.plotly_chart(fig_bar, use_container_width=True)

# ═══ TAB 4: 멀티팩터 스크리너 ════════════════════════════════════════════════
with tab4:
    if show_screen:
        st.markdown("### 🔬 멀티팩터 스크리너")
        st.caption("조회 종목에 대해 주요 팩터를 계산하여 순위를 매깁니다.")

        # 각 종목별 팩터 계산
        factor_rows = []
        for code in loaded_codes:
            df_c = df_raw[df_raw["sh7code"] == code].sort_values("date")
            if len(df_c) < 22:
                continue
            c = df_c["close"]
            ret_1m  = c.pct_change(21).iloc[-1]
            ret_3m  = c.pct_change(63).iloc[-1] if len(df_c) > 63 else np.nan
            ret_6m  = c.pct_change(126).iloc[-1] if len(df_c) > 126 else np.nan
            ret_12m = c.pct_change(252).iloc[-1] if len(df_c) > 252 else np.nan
            vol_21  = historical_volatility(c, 21, annualize=True).iloc[-1]
            rsi_val = rsi(c, 14).iloc[-1]
            mc_val  = df_c["mc"].iloc[-1] / 1e12

            factor_rows.append({
                "종목코드":       code,
                "기업명":         stock_short_label(code),
                "1M 수익률":     ret_1m,
                "3M 수익률":     ret_3m,
                "6M 수익률":     ret_6m,
                "12M 수익률":    ret_12m,
                "실현변동성(연)": vol_21,
                "RSI (14)":     rsi_val,
                "시가총액(조)":  mc_val,
            })

        if factor_rows:
            factor_df = pd.DataFrame(factor_rows).set_index("종목코드")
            _cols = ["기업명"] + [c for c in factor_df.columns if c != "기업명"]
            factor_df = factor_df[_cols]

            # 정렬 기준 선택
            sort_col = st.selectbox("정렬 기준", factor_df.columns.tolist(), index=1)
            ascending = st.checkbox("오름차순 정렬", value=False)
            factor_df = factor_df.sort_values(sort_col, ascending=ascending)

            # 포맷팅
            fmt = {
                "1M 수익률":     "{:.2%}",
                "3M 수익률":     "{:.2%}",
                "6M 수익률":     "{:.2%}",
                "12M 수익률":    "{:.2%}",
                "실현변동성(연)": "{:.2%}",
                "RSI (14)":     "{:.1f}",
                "시가총액(조)":  "{:.2f}",
            }
            styled = factor_df.style
            for col, f in fmt.items():
                if col in factor_df.columns:
                    styled = styled.format(f, subset=[col])

            # 색상 강조
            styled = styled.background_gradient(
                cmap="RdYlGn",
                subset=["1M 수익률", "3M 수익률", "6M 수익률", "12M 수익률"],
            )

            st.dataframe(styled, use_container_width=True)
            download_csv_button(
                factor_df.reset_index(),
                file_name=f"factor_screen_{start_di}_{end_di}.csv",
                label="팩터 표 CSV 다운로드",
                key="dl_factor",
            )
        else:
            st.warning("팩터 계산에 충분한 데이터가 없습니다.")
