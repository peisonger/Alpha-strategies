"""
간단 롱온리 백테스트 (이동평균 크로스 등).
"""

from __future__ import annotations

import pandas as pd


def ma_crossover_returns(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """
    fast 이평 > slow 이평일 때만 전일 종가 대비 수익률을 가져가는 단순 롱 전략.

    Parameters
    ----------
    close : 종가 시계열
    fast, slow : 이평 기간 (fast < slow 권장)
    """
    if fast >= slow:
        raise ValueError("fast는 slow보다 작아야 합니다.")
    # min_periods는 window 이하여야 함. 느린 이평이 채워지기 전에는 s가 NaN이라 (f > s)는 False.
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    long_pos = (f > s).astype(float)
    ret = close.pct_change()
    return long_pos.shift(1).fillna(0.0) * ret
