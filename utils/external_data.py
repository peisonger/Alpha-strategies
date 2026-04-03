"""
외부 공개 API 시세 스냅샷 (API 키 불필요).

- Binance: BTC/USDT 24h 티커
- Frankfurter: USD→KRW 환율

성능: Streamlit @st.cache_data 로 TTL 내 재호출 방지 (과제·보고서에서 '최적화' 근거로 사용 가능).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests
import streamlit as st


BINANCE_TICKER = "https://api.binance.com/api/v3/ticker/24hr"
FRANKFURTER = "https://api.frankfurter.app/latest"


def _fetch_binance_btcusdt() -> dict[str, Any]:
    r = requests.get(BINANCE_TICKER, params={"symbol": "BTCUSDT"}, timeout=12)
    r.raise_for_status()
    j = r.json()
    return {
        "price": float(j["lastPrice"]),
        "chg_pct_24h": float(j["priceChangePercent"]),
        "volume_usdt_billions": float(j["quoteVolume"]) / 1e9,
        "high_24h": float(j["highPrice"]),
        "low_24h": float(j["lowPrice"]),
    }


def _fetch_usd_krw() -> float:
    r = requests.get(FRANKFURTER, params={"from": "USD", "to": "KRW"}, timeout=12)
    r.raise_for_status()
    data = r.json()
    return float(data["rates"]["KRW"])


@st.cache_data(ttl=90, show_spinner=False)
def get_live_market_snapshot() -> dict[str, Any]:
    """
    외부 API 병합 결과. 일부 실패 시 ok 플래그로 구분.

    Returns
    -------
    dict
        btc: {...} | None
        usd_krw: float | None
        fetched_at: ISO 시간 (UTC)
        errors: list[str]
    """
    out: dict[str, Any] = {
        "btc": None,
        "usd_krw": None,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "errors": [],
    }
    try:
        out["btc"] = _fetch_binance_btcusdt()
    except Exception as e:
        out["errors"].append(f"Binance BTC: {e!s}")
    try:
        out["usd_krw"] = _fetch_usd_krw()
    except Exception as e:
        out["errors"].append(f"환율(Frankfurter): {e!s}")
    return out


def clear_external_cache() -> None:
    """사이드바 '새로고침' 등에서 사용."""
    get_live_market_snapshot.clear()
