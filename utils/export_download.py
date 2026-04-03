"""Streamlit용 CSV 다운로드 헬퍼."""

from __future__ import annotations

import streamlit as st
import pandas as pd


def download_csv_button(
    df: pd.DataFrame,
    file_name: str,
    label: str = "CSV로 다운로드",
    key: str | None = None,
) -> None:
    """UTF-8 BOM CSV 다운로드 버튼."""
    data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label,
        data=data,
        file_name=file_name,
        mime="text/csv; charset=utf-8",
        key=key,
    )
