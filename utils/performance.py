"""
utils/performance.py
─────────────────────
포트폴리오 / 전략 성과 지표 계산 모듈.

포함 메트릭:
    - 총 수익률 / CAGR
    - Sharpe / Sortino / Calmar
    - Max Drawdown / Drawdown 기간
    - VaR / CVaR (95%, 99%)
    - Hit Rate / Profit Factor
    - 월별 수익률 히트맵용 피벗
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


def summary_stats(
    returns: pd.Series,
    freq: int = 252,
    rf: float = 0.0,
    label: str = "전략",
) -> dict:
    """
    수익률 시계열로 주요 성과 지표 계산.

    Parameters
    ----------
    returns : pd.Series  (일간 수익률, 예: pct_change())
    freq    : int        (연간 거래일 수 — 일봉=252, 1h=252*24)
    rf      : float      (무위험 수익률, 연간)
    label   : str        (전략 이름)

    Returns
    -------
    dict   키별 수치
    """
    r = returns.dropna()

    if len(r) < 2:
        return {}

    # 누적 수익률
    cum = (1 + r).cumprod()
    total_return = cum.iloc[-1] - 1

    # CAGR
    n_years = len(r) / freq
    cagr    = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    # 변동성 (연환산)
    ann_vol = r.std() * np.sqrt(freq)

    # Sharpe
    excess  = r - rf / freq
    sharpe  = (excess.mean() / excess.std()) * np.sqrt(freq) if excess.std() > 0 else np.nan

    # Sortino (하방 변동성만)
    downside = r[r < 0].std() * np.sqrt(freq)
    sortino  = (r.mean() * freq - rf) / downside if downside > 0 else np.nan

    # 최대 낙폭
    roll_max = cum.cummax()
    dd       = (cum - roll_max) / roll_max
    max_dd   = dd.min()

    # 낙폭 지속 기간 (영업일)
    in_dd       = dd < 0
    dd_start    = None
    max_dd_days = 0
    current_days = 0
    for val in in_dd:
        if val:
            current_days += 1
            max_dd_days = max(max_dd_days, current_days)
        else:
            current_days = 0

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    # VaR / CVaR
    var_95  = np.percentile(r, 5)
    cvar_95 = r[r <= var_95].mean()
    var_99  = np.percentile(r, 1)
    cvar_99 = r[r <= var_99].mean()

    # 승률 / Profit Factor
    wins  = r[r > 0]
    loses = r[r < 0]
    hit_rate      = len(wins) / len(r) if len(r) > 0 else np.nan
    profit_factor = abs(wins.sum() / loses.sum()) if loses.sum() != 0 else np.nan

    return {
        "label":          label,
        "total_return":   total_return,
        "cagr":           cagr,
        "ann_vol":        ann_vol,
        "sharpe":         sharpe,
        "sortino":        sortino,
        "calmar":         calmar,
        "max_drawdown":   max_dd,
        "max_dd_days":    max_dd_days,
        "var_95":         var_95,
        "cvar_95":        cvar_95,
        "var_99":         var_99,
        "cvar_99":        cvar_99,
        "hit_rate":       hit_rate,
        "profit_factor":  profit_factor,
        "n_trades":       len(r),
    }


def format_stats_df(stats: dict) -> pd.DataFrame:
    """summary_stats 결과를 보기 좋은 DataFrame으로 변환."""
    fmt_map = {
        "total_return":  ("{:.2%}", "총 수익률"),
        "cagr":          ("{:.2%}", "연평균 복리 수익률 (CAGR)"),
        "ann_vol":       ("{:.2%}", "연간 변동성"),
        "sharpe":        ("{:.2f}", "샤프 비율"),
        "sortino":       ("{:.2f}", "소르티노 비율"),
        "calmar":        ("{:.2f}", "칼마 비율"),
        "max_drawdown":  ("{:.2%}", "최대 낙폭"),
        "max_dd_days":   ("{:.0f}일", "최대 낙폭 지속"),
        "var_95":        ("{:.2%}", "VaR (95%)"),
        "cvar_95":       ("{:.2%}", "CVaR (95%)"),
        "hit_rate":      ("{:.2%}", "승률"),
        "profit_factor": ("{:.2f}", "손익비"),
    }
    rows = []
    for key, (fmt, label) in fmt_map.items():
        val = stats.get(key, np.nan)
        try:
            rows.append({"지표": label, "값": fmt.format(val)})
        except Exception:
            rows.append({"지표": label, "값": str(val)})
    return pd.DataFrame(rows)


def monthly_returns_pivot(returns: pd.Series) -> pd.DataFrame:
    """
    일간 수익률 → 월별 수익률 피벗 테이블 (히트맵용).

    Returns
    -------
    pd.DataFrame  index=Year, columns=Month(1~12)
    """
    monthly = (1 + returns).resample("ME").prod() - 1
    df = monthly.to_frame("ret")
    df["year"]  = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot.columns = [f"{int(c)}월" for c in pivot.columns]
    return pivot


def rolling_metrics(
    returns: pd.Series,
    window: int = 252,
    freq: int = 252,
) -> pd.DataFrame:
    """
    롤링 Sharpe / 변동성 / 수익률.

    Returns
    -------
    pd.DataFrame  columns: [roll_sharpe, roll_vol, roll_ret]
    """
    roll_mean = returns.rolling(window).mean()
    roll_std  = returns.rolling(window).std()
    roll_sharpe = (roll_mean / roll_std) * np.sqrt(freq)
    roll_vol    = roll_std * np.sqrt(freq)
    roll_ret    = (1 + returns).rolling(window).apply(np.prod, raw=True) - 1
    return pd.DataFrame({
        "roll_sharpe": roll_sharpe,
        "roll_vol":    roll_vol,
        "roll_ret":    roll_ret,
    })


def compare_strategies(*args: tuple[str, pd.Series], freq: int = 252) -> pd.DataFrame:
    """
    여러 전략 성과 비교 테이블.

    Usage
    -----
    compare_strategies(
        ("BTC Buy&Hold", btc_returns),
        ("K200F MOM",    k200_returns),
    )
    """
    rows = []
    for label, ret in args:
        s = summary_stats(ret, freq=freq, label=label)
        rows.append(s)
    df = pd.DataFrame(rows).set_index("label")
    pct_cols = ["total_return","cagr","ann_vol","max_drawdown","var_95","cvar_95","hit_rate"]
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
    float_cols = ["sharpe","sortino","calmar","profit_factor"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    return df
