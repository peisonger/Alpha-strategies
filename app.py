"""
Quant Dashboard — Main Entry Point
Multi-asset: BTC/USDT · Korean Derivatives · Stock Universe
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from utils.external_data import clear_external_cache, get_live_market_snapshot

st.set_page_config(
    page_title="퀀트 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0d1117; }
    [data-testid="stSidebar"] * { color: #e6edf3 !important; }
    [data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
    }
    /* 어두운 카드 + Streamlit 기본(어두운 글자색) → 대비 붕괴 방지 */
    [data-testid="stMetric"] label { color: #8b949e !important; }
    [data-testid="stMetricLabel"] { color: #8b949e !important; }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700;
        color: #f0f6fc !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 퀀트 대시보드")
    st.markdown("---")
    st.markdown("### 메뉴")
    st.page_link("app.py",                    label="홈")
    st.page_link("pages/1_BTC_Trading.py",    label="BTC 거래")
    st.page_link("pages/2_KR_Market.py",      label="한국 시장")
    st.page_link("pages/3_Derivatives.py",    label="파생상품")
    st.page_link("pages/4_Stock_Universe.py", label="전종목 스크리너")
    st.markdown("---")
    st.caption("데이터: 2000–2025 · 멀티 자산")
    if st.button("외부 시세 캐시 비우기", help="Binance·환율 API를 다시 호출합니다."):
        clear_external_cache()
        st.rerun()

# ─── Home Page ────────────────────────────────────────────────────────────────
st.markdown("# 퀀트 분석 대시보드")
st.markdown("**멀티 자산 퀀트 리서치** — BTC/USDT · 한국 선물 · 옵션 · 전종목")
st.markdown("---")

# ─── 외부 API 시세 (캐시 TTL 90초 — API 부하·응답 속도 균형) ─────────────────
snap = get_live_market_snapshot()
st.markdown("### 외부 시세 스냅샷 (실시간)")
if snap["errors"] and not snap["btc"] and snap["usd_krw"] is None:
    st.warning(
        "외부 API를 불러오지 못했습니다. 네트워크·방화벽을 확인하거나 잠시 후 다시 시도하세요.\n\n"
        + "\n".join(snap["errors"])
    )
else:
    if snap["btc"] and snap["usd_krw"] is not None:
        ec1, ec2, ec3, ec4, ec5 = st.columns(5)
        b = snap["btc"]
        ec1.metric(
            "BTC/USDT (Binance)",
            f"${b['price']:,.2f}",
            f"{b['chg_pct_24h']:+.2f}% (24h)",
        )
        ec2.metric("24h 거래대금 (USDT)", f"{b['volume_usdt_billions']:.2f}B")
        ec3.metric("24h 고가", f"${b['high_24h']:,.0f}")
        ec4.metric("24h 저가", f"${b['low_24h']:,.0f}")
        ec5.metric("USD/KRW (Frankfurter)", f"{snap['usd_krw']:,.2f}원/$")
    elif snap["btc"]:
        b = snap["btc"]
        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric(
            "BTC/USDT (Binance)",
            f"${b['price']:,.2f}",
            f"{b['chg_pct_24h']:+.2f}% (24h)",
        )
        ec2.metric("24h 거래대금 (USDT)", f"{b['volume_usdt_billions']:.2f}B")
        ec3.metric("24h 고가", f"${b['high_24h']:,.0f}")
        ec4.metric("24h 저가", f"${b['low_24h']:,.0f}")
    elif snap["usd_krw"] is not None:
        st.metric("USD/KRW (Frankfurter)", f"{snap['usd_krw']:,.2f}원/$")
    if snap["errors"]:
        for err in snap["errors"]:
            st.caption(f"일부 소스 실패: {err}")
st.caption(
    f"수집 시각(서버 기준 UTC): `{snap['fetched_at']}` · "
    "`@st.cache_data(ttl=90)` 로 동일 세션 내 반복 호출을 줄입니다."
)
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("BTC OHLCV", "2017–2025", "5m / 1h / 6h / 1d / 1w")
with col2:
    st.metric("한국 지수", "1980–2023", "KOSPI · KOSDAQ · VIX · KRW")
with col3:
    st.metric("파생상품", "2000–2024", "K200F · K200MF · KQ150F · 옵션")
with col4:
    st.metric("전종목", "2010–2023", "조정주가 780만 행")

st.markdown("---")
st.markdown("### 모듈 안내")
c1, c2 = st.columns(2)
with c1:
    st.info(
        "**BTC 멀티 타임프레임**\n\n"
        "5m · 1h · 6h · 1d · 1w 캔들 · RSI · MACD · 볼린저 · 거래량 프로파일\n"
        "누적 손익 낙폭 차트 · 이평 크로스 간단 백테스트 vs 매수·보유\n"
        "OHLCV·수익률 CSV 저장"
    )
    st.info(
        "**파생상품**\n\n"
        "K200 / K200MF / KQ150 선물 · 스프레드\n"
        "옵션 체인 · 풋/콜 비율 · IV 스마일(근사)\n"
        "선물·옵션 CSV 내보내기"
    )
with c2:
    st.info(
        "**한국 매크로**\n\n"
        "KOSPI · KOSDAQ · KOSPI200 · K-VIX · USD/KRW\n"
        "상관 히트맵 · 롤링 상관 · 월별 수익률\n"
        "60일 실현변동성 · 지수 시계열 CSV 저장"
    )
    st.info(
        "**전종목 스크리너**\n\n"
        "수정주가 Parquet 고속 필터 · 외국인·기관 수급 · 시가총액\n"
        "1M·3M·6M·12M 수익률 · 변동성 · RSI · 멀티팩터 순위\n"
        "조회 결과·팩터 표 CSV 저장"
    )

st.markdown("---")
st.caption("Streamlit · Plotly · Pandas | @st.cache_data · Parquet · 조기 필터")
