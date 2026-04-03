"""
utils/charts.py
────────────────
Plotly 차트 팩토리 — 재사용 가능한 고품질 차트 컴포넌트.

다크 테마 기반 · WebGL 모드 지원.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── 테마 ────────────────────────────────────────────────────────────────────

DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
BORDER     = "#30363d"
TEXT_COLOR = "#e6edf3"
MUTED      = "#8b949e"
GREEN      = "#3fb950"
RED        = "#f85149"
BLUE       = "#58a6ff"
ORANGE     = "#d29922"
PURPLE     = "#bc8cff"

LAYOUT_BASE = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT_COLOR, size=12),
    margin=dict(l=60, r=20, t=40, b=40),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, showline=True, linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, showline=True, linecolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
    hoverlabel=dict(bgcolor=PANEL_BG, bordercolor=BORDER, font_color=TEXT_COLOR),
)


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**LAYOUT_BASE)
    return fig


# ─── 캔들스틱 차트 ──────────────────────────────────────────────────────────

def candlestick_chart(
    df: pd.DataFrame,
    date_col: str = "datetime",
    title: str = "가격",
    show_volume: bool = True,
    indicators: list[dict] | None = None,
) -> go.Figure:
    """
    OHLCV 캔들스틱 + 거래량 서브플롯.

    Parameters
    ----------
    df         : OHLCV DataFrame
    date_col   : 날짜 컬럼명
    title      : 차트 제목
    show_volume: 거래량 서브플롯 표시 여부
    indicators : [{"name": "SMA20", "data": pd.Series, "color": "#58a6ff"}, ...]

    Returns
    -------
    go.Figure
    """
    close_col = "Close" if "Close" in df.columns else "close"
    open_col  = "Open"  if "Open"  in df.columns else "open"
    high_col  = "High"  if "High"  in df.columns else "high"
    low_col   = "Low"   if "Low"   in df.columns else "low"
    vol_col   = "Volume" if "Volume" in df.columns else "vol"

    rows = 2 if show_volume else 1
    row_heights = [0.75, 0.25] if show_volume else [1.0]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.02,
    )

    # 캔들스틱
    fig.add_trace(go.Candlestick(
        x=df[date_col],
        open=df[open_col], high=df[high_col],
        low=df[low_col],   close=df[close_col],
        increasing_line_color=GREEN,
        decreasing_line_color=RED,
        name="가격",
        showlegend=False,
    ), row=1, col=1)

    # 오버레이 지표 (SMA, EMA, BB 등)
    if indicators:
        colors = [BLUE, ORANGE, PURPLE, "#00d4aa", "#ff9500"]
        for i, ind in enumerate(indicators):
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=ind["data"],
                mode="lines",
                line=dict(color=ind.get("color", colors[i % len(colors)]), width=1.2),
                name=ind["name"],
                hovertemplate="%{y:.2f}",
            ), row=1, col=1)

    # 거래량
    if show_volume:
        colors_vol = [GREEN if df[close_col].iloc[i] >= df[open_col].iloc[i] else RED
                      for i in range(len(df))]
        fig.add_trace(go.Bar(
            x=df[date_col],
            y=df[vol_col],
            marker_color=colors_vol,
            name="거래량",
            showlegend=False,
            opacity=0.7,
        ), row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR)),
        xaxis_rangeslider_visible=False,
        height=600,
        **LAYOUT_BASE,
    )
    fig.update_yaxes(title_text="가격", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="거래량", row=2, col=1)

    return fig


# ─── RSI 차트 ────────────────────────────────────────────────────────────────

def rsi_chart(rsi_series: pd.Series, date_series: pd.Series, title: str = "RSI (14)") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=date_series, y=rsi_series, mode="lines",
        line=dict(color=PURPLE, width=1.5), name="RSI",
    ))
    fig.add_hline(y=70, line=dict(color=RED, dash="dash", width=1), annotation_text="과매수(70)")
    fig.add_hline(y=30, line=dict(color=GREEN, dash="dash", width=1), annotation_text="과매도(30)")
    fig.add_hrect(y0=70, y1=100, fillcolor=RED, opacity=0.05, line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor=GREEN, opacity=0.05, line_width=0)
    fig.update_layout(title=title, height=250, **LAYOUT_BASE)
    fig.update_yaxes(range=[0, 100])
    return fig


# ─── MACD 차트 ───────────────────────────────────────────────────────────────

def macd_chart(macd_df: pd.DataFrame, date_series: pd.Series, title: str = "MACD") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=date_series, y=macd_df["histogram"],
        marker_color=[GREEN if v >= 0 else RED for v in macd_df["histogram"]],
        name="히스토그램", opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=date_series, y=macd_df["macd_line"],
        line=dict(color=BLUE, width=1.5), name="MACD",
    ))
    fig.add_trace(go.Scatter(
        x=date_series, y=macd_df["signal_line"],
        line=dict(color=ORANGE, width=1.5), name="시그널",
    ))
    fig.update_layout(title=title, height=250, **LAYOUT_BASE)
    return fig


# ─── 수익률 누적 차트 ────────────────────────────────────────────────────────

def cumulative_returns_chart(
    series_dict: dict[str, pd.Series],  # {"전략명": 수익률 시리즈}
    title: str = "누적 수익률",
) -> go.Figure:
    colors = [BLUE, GREEN, ORANGE, PURPLE, RED]
    fig = go.Figure()
    for i, (name, ret) in enumerate(series_dict.items()):
        cum = (1 + ret.dropna()).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum * 100,
            mode="lines", name=name,
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate="%{y:.1f}%",
        ))
    fig.update_layout(
        title=title,
        yaxis_title="수익률 (%)",
        height=400,
        **LAYOUT_BASE,
    )
    return fig


# ─── 상관관계 히트맵 ─────────────────────────────────────────────────────────

def correlation_heatmap(corr_df: pd.DataFrame, title: str = "상관관계 히트맵") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns.tolist(),
        y=corr_df.index.tolist(),
        colorscale=[
            [0.0, "#f85149"], [0.5, PANEL_BG], [1.0, "#3fb950"]
        ],
        zmin=-1, zmax=1,
        text=corr_df.round(2).values,
        texttemplate="%{text}",
        hovertemplate="X: %{x}<br>Y: %{y}<br>상관: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(title=title, height=400, **LAYOUT_BASE)
    return fig


# ─── Volume Profile 수평 바 차트 ─────────────────────────────────────────────

def volume_profile_chart(vp_df: pd.DataFrame, title: str = "거래량 프로파일") -> go.Figure:
    fig = go.Figure(go.Bar(
        x=vp_df["volume"],
        y=vp_df["price_mid"],
        orientation="h",
        marker_color=BLUE,
        opacity=0.7,
        name="거래량",
        hovertemplate="가격: %{y:.0f}<br>거래량: %{x:,.0f}<extra></extra>",
    ))
    # POC (Point of Control) 표시
    poc_idx = vp_df["volume"].idxmax()
    poc_price = vp_df.loc[poc_idx, "price_mid"]
    fig.add_hline(y=poc_price, line=dict(color=ORANGE, dash="dash", width=1.5),
                  annotation_text=f"집중가: {poc_price:,.0f}")
    fig.update_layout(title=title, height=400, **LAYOUT_BASE,
                      xaxis_title="거래량", yaxis_title="가격")
    return fig


# ─── 월별 수익률 히트맵 ───────────────────────────────────────────────────────

def monthly_returns_heatmap(pivot: pd.DataFrame, title: str = "월별 수익률") -> go.Figure:
    text = pivot.applymap(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "")
    fig = go.Figure(go.Heatmap(
        z=pivot.values * 100,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, RED], [0.5, PANEL_BG], [1, GREEN]],
        text=text.values,
        texttemplate="%{text}",
        hovertemplate="월: %{x}<br>연도: %{y}<br>수익률: %{z:.2f}%<extra></extra>",
        zmid=0,
    ))
    fig.update_layout(title=title, height=350, **LAYOUT_BASE)
    return fig
