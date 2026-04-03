"""
utils/data_loader.py
────────────────────
성능 최적화 핵심 모듈.

최적화 전략:
  1. CSV → Parquet 1회 변환 (10x 속도 향상)
  2. @st.cache_data + TTL 로 재연산 방지
  3. 컬럼 프루닝 — 필요한 컬럼만 읽기
  4. 날짜 범위 필터링 (Predicate Pushdown)
  5. st.session_state 로 페이지 전환 간 데이터 유지
"""

from __future__ import annotations

import time
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"          # Parquet 캐시 저장소
UPLOAD_DIR = Path(__file__).parent.parent / "data"  # 원본 CSV 위치

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. BTC OHLCV  (5m · 1h · 6h · 1d · 1w)
# ══════════════════════════════════════════════════════════════════════════════

BTC_FILES = {
    "5m":  "btcusdt_5m.csv",       # zip 내부 파일명
    "1h":  "btcusdt_1h.csv",
    "6h":  "btcusdt_6h.csv",
    "1d":  "btcusdt_1d.csv",
    "1w":  "btcusdt_1w.csv",
}

def _btc_parquet_path(tf: str) -> Path:
    return DATA_DIR / f"btc_{tf}.parquet"


def _convert_btc_to_parquet(tf: str) -> Path:
    """CSV → Parquet 1회 변환 (이후 캐시 사용)."""
    pq_path = _btc_parquet_path(tf)
    if pq_path.exists():
        return pq_path

    with st.spinner(f"📦 BTC {tf} 데이터 Parquet 변환 중…"):
        t0 = time.perf_counter()

        if tf == "5m":
            import zipfile
            zip_path = UPLOAD_DIR / "btcusdt_5m_csv.zip"
            csv_path = UPLOAD_DIR / BTC_FILES["5m"]
            if zip_path.exists():
                with zipfile.ZipFile(zip_path) as z:
                    with z.open("btcusdt_5m.csv") as f:
                        df = pd.read_csv(f)
            elif csv_path.exists():
                df = pd.read_csv(csv_path)
            else:
                raise FileNotFoundError(
                    f"BTC 5m: {zip_path.name} 또는 {csv_path.name} 이(가) "
                    f"{UPLOAD_DIR} 에 필요합니다."
                )
        else:
            csv_path = UPLOAD_DIR / BTC_FILES[tf]
            df = pd.read_csv(csv_path)

        # 타입 최적화
        df["dt_"] = pd.to_datetime(df["dt_"], utc=True)
        df = df.rename(columns={"dt_": "datetime"})
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].astype("float32")
        df["Volume"] = df["Volume"].astype("float32")
        df = df.sort_values("datetime").reset_index(drop=True)

        df.to_parquet(pq_path, index=False)
        elapsed = time.perf_counter() - t0
        st.toast(f"✅ BTC {tf} 변환 완료 ({elapsed:.1f}s)", icon="📦")

    return pq_path


@st.cache_data(ttl=3600, show_spinner=False)
def load_btc(
    tf: str = "1d",
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> pd.DataFrame:
    """
    BTC OHLCV 로더.

    Parameters
    ----------
    tf : str
        타임프레임 — '5m' | '1h' | '6h' | '1d' | '1w'
    start_date : str, optional
        'YYYY-MM-DD' 형식 시작일
    end_date : str, optional
        'YYYY-MM-DD' 형식 종료일

    Returns
    -------
    pd.DataFrame
        columns: datetime, Open, High, Low, Close, Volume
    """
    pq_path = _convert_btc_to_parquet(tf)

    df = pd.read_parquet(pq_path)

    if start_date:
        df = df[df["datetime"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        df = df[df["datetime"] <= pd.Timestamp(end_date, tz="UTC")]

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 한국 지수 (KOSPI · KOSDAQ · KOSPI200 · K-VIX · USD/KRW)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_kr_index(
    codes: Optional[list[str]] = None,
    start_date: Optional[str]  = None,
    end_date:   Optional[str]  = None,
) -> pd.DataFrame:
    """
    한국 지수 로더.

    Parameters
    ----------
    codes : list of str, optional
        ['KOSPI', 'KOSDAQ', 'KOSPI200', 'K-VIX', 'F-USDKRW']
        None → 전체 반환
    start_date / end_date : str, optional
        'YYYYMMDD' 또는 'YYYY-MM-DD'

    Returns
    -------
    pd.DataFrame  (wide format, index=date)
    """
    pq_path = DATA_DIR / "kr_index.parquet"

    if not pq_path.exists():
        with st.spinner("📦 한국 지수 Parquet 변환 중…"):
            df = pd.read_csv(UPLOAD_DIR / "INDEX_KR.csv")
            df["date"] = pd.to_datetime(df["dateint"].astype(str), format="%Y%m%d")
            df["close"] = df["close"].astype("float32")
            df.to_parquet(pq_path, index=False)

    df = pd.read_parquet(pq_path)

    if codes:
        df = df[df["indxcode"].isin(codes)]
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    return df.reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_kr_index_wide(
    codes: Optional[list[str]] = None,
    start_date: Optional[str]  = None,
    end_date:   Optional[str]  = None,
) -> pd.DataFrame:
    """지수별 컬럼으로 피벗한 wide-format 반환."""
    df = load_kr_index(codes=codes, start_date=start_date, end_date=end_date)
    wide = df.pivot_table(index="date", columns="indxcode", values="close")
    wide = wide.sort_index()
    return wide


# ══════════════════════════════════════════════════════════════════════════════
# 3. 한국 선물 (K200F · K200MF · KQ150F)
# ══════════════════════════════════════════════════════════════════════════════

FUTURES_FILES = {
    "K200F":   "K200_F__dailycd_cont.20240201.csv",
    "K200MF":  "K200_MF__dailycd_cont.20240201.csv",
    "KQ150F":  "KQ150_F__dailycd_cont.20240201.csv",
}


@st.cache_data(ttl=3600, show_spinner=False)
def load_futures(
    symbol: str = "K200F",
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> pd.DataFrame:
    """
    한국 선물 연속 일봉 로더.

    Parameters
    ----------
    symbol : 'K200F' | 'K200MF' | 'KQ150F'
    """
    pq_path = DATA_DIR / f"{symbol}.parquet"

    if not pq_path.exists():
        with st.spinner(f"📦 {symbol} Parquet 변환 중…"):
            df = pd.read_csv(UPLOAD_DIR / FUTURES_FILES[symbol])
            df.columns = df.columns.str.strip().str.lstrip("\ufeff")
            df["date"] = pd.to_datetime(df["date_"].str.strip(), format="%Y/%m/%d")
            df = df.drop(columns=["date_"])
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype("float32")
            df = df.sort_values("date").reset_index(drop=True)
            df.to_parquet(pq_path, index=False)

    df = pd.read_parquet(pq_path)

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    return df.reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_all_futures(
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """K200F / K200MF / KQ150F 세 가지 선물 한번에 로드."""
    return {
        sym: load_futures(sym, start_date=start_date, end_date=end_date)
        for sym in FUTURES_FILES
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. KOSPI200 옵션
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_options(
    year_month: Optional[str] = None,  # 'YYYYMM'
    direction:  Optional[int] = None,  # 1=콜, -1=풋
    strike_min: Optional[float] = None,
    strike_max: Optional[float] = None,
) -> pd.DataFrame:
    """
    KOSPI200 옵션 데이터 로더 (22만 행).

    Parameters
    ----------
    year_month : str, optional
        만기 월 필터 — 예: '201903'
    direction : int, optional
        1 (콜) 또는 -1 (풋)
    strike_min / strike_max : float, optional
        행사가 범위 필터

    Returns
    -------
    pd.DataFrame
        columns: stock_code, date, f_ym, odir, f_prc, open, high, low, close, vol, trd_val, ...
    """
    pq_path = DATA_DIR / "kospi200_option.parquet"

    if not pq_path.exists():
        with st.spinner("KOSPI200 옵션 Parquet 변환 중… (22만 행)"):
            df = pd.read_csv(UPLOAD_DIR / "KOSPI200_OPTION.csv")
            df["date"] = pd.to_datetime(df["dateint"].astype(str), format="%Y%m%d")
            df["f_ym"] = df["f_ym"].astype(str)
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype("float32")
            df["vol"]      = df["vol"].astype("int32")
            df["trd_val"]  = df["trd_val"].astype("int64")
            df.to_parquet(pq_path, index=False)

    df = pd.read_parquet(pq_path)

    # 필터링
    if year_month:
        df = df[df["f_ym"] == year_month]
    if direction is not None:
        df = df[df["odir"] == direction]
    if strike_min is not None:
        df = df[df["f_prc"] >= strike_min]
    if strike_max is not None:
        df = df[df["f_prc"] <= strike_max]

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. 전종목 주가 (adjchart / rawchart Parquet — 780만 행)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_universe(
    parquet_path: str,
    columns: Optional[list[str]]   = None,
    codes:   Optional[list[str]]   = None,
    start_dateint: Optional[int]   = None,
    end_dateint:   Optional[int]   = None,
) -> pd.DataFrame:
    """
    수정주가/원시주가 Parquet 로더 (780만 행 고성능).

    Parameters
    ----------
    parquet_path : str
        adjchart_df 또는 rawchart_df Parquet 파일 경로
    columns : list of str, optional
        필요한 컬럼만 선택 (컬럼 프루닝 — 속도 핵심)
    codes : list of str, optional
        종목코드 필터 — 예: ['A005930', 'A000660']
    start_dateint / end_dateint : int, optional
        날짜 범위 필터 — 예: 20200101

    Returns
    -------
    pd.DataFrame
    """
    read_cols = columns or [
        "dateint", "sh7code", "open", "high", "low", "close",
        "vol", "trd_val", "mc", "org_netvol", "frg_belong_sh"
    ]

    df = pd.read_parquet(parquet_path, columns=read_cols)

    # Predicate Pushdown (Pandas 레벨)
    if codes:
        df = df[df["sh7code"].isin(codes)]
    if start_dateint:
        df = df[df["dateint"] >= start_dateint]
    if end_dateint:
        df = df[df["dateint"] <= end_dateint]

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6. 세션 상태 헬퍼 (페이지 전환 간 데이터 유지)
# ══════════════════════════════════════════════════════════════════════════════

def get_session(key: str, loader_fn, *args, **kwargs):
    """
    st.session_state 기반 캐시 레이어.
    동일 키가 존재하면 재연산 없이 반환.

    Usage
    -----
    df = get_session("btc_1d", load_btc, tf="1d")
    """
    if key not in st.session_state:
        st.session_state[key] = loader_fn(*args, **kwargs)
    return st.session_state[key]


def clear_session_cache():
    """모든 세션 데이터 초기화."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 7. 성능 벤치마크 유틸
# ══════════════════════════════════════════════════════════════════════════════

def benchmark(fn, *args, label: str = "작업", **kwargs):
    """함수 실행 시간 측정 후 (result, elapsed_ms) 반환."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    st.sidebar.caption(f"{label}: {elapsed_ms:.0f}ms")
    return result, elapsed_ms
