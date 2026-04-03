"""
utils/indicators.py
────────────────────
기술적 지표 계산 모듈 (Pandas 순수 구현 — 외부 라이브러리 불필요).

포함 지표:
    - SMA / EMA
    - RSI
    - MACD
    - Bollinger Bands
    - ATR
    - Volume Profile (가격대별 거래량)
    - Rolling Correlation
    - Drawdown
"""

from __future__ import annotations
import pandas as pd
import numpy as np


# ─── 이동평균 ────────────────────────────────────────────────────────────────

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ─── RSI ─────────────────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_l  = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─── MACD ────────────────────────────────────────────────────────────────────

def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Returns
    -------
    pd.DataFrame  columns: [macd_line, signal_line, histogram]
    """
    fast_ema   = ema(series, fast)
    slow_ema   = ema(series, slow)
    macd_line  = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return pd.DataFrame({
        "macd_line":   macd_line,
        "signal_line": signal_line,
        "histogram":   histogram,
    })


# ─── 볼린저 밴드 ──────────────────────────────────────────────────────────────

def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    Returns
    -------
    pd.DataFrame  columns: [bb_mid, bb_upper, bb_lower, bb_width, bb_pct]
    """
    mid     = sma(series, window)
    std     = series.rolling(window).std()
    upper   = mid + num_std * std
    lower   = mid - num_std * std
    width   = (upper - lower) / mid
    pct     = (series - lower) / (upper - lower)
    return pd.DataFrame({
        "bb_mid":   mid,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_width": width,
        "bb_pct":   pct,
    })


# ─── ATR ─────────────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """True Range 기반 ATR. df 에 high/low/close 컬럼 필요."""
    h   = df["High"] if "High" in df.columns else df["high"]
    l   = df["Low"]  if "Low"  in df.columns else df["low"]
    c   = df["Close"] if "Close" in df.columns else df["close"]
    prev_c = c.shift(1)
    tr  = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


# ─── Volume Profile ──────────────────────────────────────────────────────────

def volume_profile(
    df: pd.DataFrame,
    bins: int = 30,
) -> pd.DataFrame:
    """
    가격대별 누적 거래량 (Volume Profile).

    Parameters
    ----------
    df  : OHLCV DataFrame (Close/close, Volume/vol 컬럼 필요)
    bins: 가격 구간 수

    Returns
    -------
    pd.DataFrame  columns: [price_low, price_high, price_mid, volume]
    """
    close_col = "Close" if "Close" in df.columns else "close"
    vol_col   = "Volume" if "Volume" in df.columns else "vol"

    price_min = df[close_col].min()
    price_max = df[close_col].max()
    edges     = np.linspace(price_min, price_max, bins + 1)

    vols = []
    for i in range(len(edges) - 1):
        mask = (df[close_col] >= edges[i]) & (df[close_col] < edges[i + 1])
        vols.append(df.loc[mask, vol_col].sum())

    return pd.DataFrame({
        "price_low":  edges[:-1],
        "price_high": edges[1:],
        "price_mid":  (edges[:-1] + edges[1:]) / 2,
        "volume":     vols,
    })


# ─── 롤링 상관관계 ───────────────────────────────────────────────────────────

def rolling_corr(s1: pd.Series, s2: pd.Series, window: int = 60) -> pd.Series:
    return s1.rolling(window).corr(s2)


# ─── 수익률 & 드로다운 ────────────────────────────────────────────────────────

def log_returns(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1))


def pct_returns(series: pd.Series) -> pd.Series:
    return series.pct_change()


def drawdown(series: pd.Series) -> pd.Series:
    """최고점 대비 낙폭 (0 ~ -1 범위)."""
    roll_max = series.cummax()
    return (series - roll_max) / roll_max


def max_drawdown(series: pd.Series) -> float:
    return drawdown(series).min()


def cagr(series: pd.Series, freq: int = 252) -> float:
    """연환산 수익률. freq: 일봉=252, 시간봉=252*24."""
    total  = series.iloc[-1] / series.iloc[0]
    n_periods = len(series) / freq
    return total ** (1 / n_periods) - 1


def sharpe(returns: pd.Series, freq: int = 252, rf: float = 0.0) -> float:
    excess = returns - rf / freq
    return (excess.mean() / excess.std()) * np.sqrt(freq)


# ─── 내재변동성 근사 (옵션용) ────────────────────────────────────────────────

def historical_volatility(series: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series:
    """실현 변동성 (Realized Volatility)."""
    rv = log_returns(series).rolling(window).std()
    if annualize:
        rv = rv * np.sqrt(252)
    return rv


# ─── 편의 함수: OHLCV에 모든 지표 한번에 추가 ────────────────────────────────

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV DataFrame에 주요 지표를 모두 추가하여 반환.
    컬럼명: Open/High/Low/Close/Volume (대문자) 또는 소문자 모두 지원.
    """
    close_col = "Close" if "Close" in df.columns else "close"
    c = df[close_col]

    out = df.copy()

    # 이동평균
    for w in [20, 50, 200]:
        out[f"sma_{w}"] = sma(c, w)
    out["ema_12"] = ema(c, 12)
    out["ema_26"] = ema(c, 26)

    # 모멘텀
    out["rsi_14"] = rsi(c, 14)

    # MACD
    macd_df = macd(c)
    out = pd.concat([out, macd_df], axis=1)

    # 볼린저 밴드
    bb_df = bollinger_bands(c)
    out = pd.concat([out, bb_df], axis=1)

    # ATR
    out["atr_14"] = atr(df, 14)

    # 변동성
    out["hist_vol_21"] = historical_volatility(c, 21)

    # 수익률
    out["pct_ret"] = pct_returns(c)
    out["log_ret"] = log_returns(c)
    out["drawdown"] = drawdown(c)

    return out
