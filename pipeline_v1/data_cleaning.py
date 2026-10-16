"""Load and clean AQI time-series data."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import create_engine

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings
from prediction_data_process import read_database_observations

FEATURES = ["O3", "PM2.5", "PM10", "CO", "SO2", "NO2"]


def load_and_clean_data(
    *,
    station: str = "木柵站",
    start: str = "2025-01-01",
    end: str = "2025-12-31",
) -> tuple[pd.DataFrame, list[str]]:
    """讀取資料庫逐時觀測；start/end 為台灣日期，包含結束日全天。"""
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    if (pd.isna(start_date) or pd.isna(end_date)
            or start_date.tzinfo is not None or end_date.tzinfo is not None
            or start_date != start_date.normalize() or end_date != end_date.normalize()
            or start_date > end_date):
        raise ValueError("start/end 必須是有效的無時區日期，且 start 不晚於 end")
    station = station.strip()
    if not station:
        raise ValueError("station 不可為空白")
    # 共用讀取函式包含兩端，因此將結束時間設成結束日最後一微秒。
    end_time = end_date + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            data = read_database_observations(connection, station, start_date, end_time)
    finally:
        engine.dispose()
    if data.empty:
        raise ValueError(f"測站「{station}」在 {start_date.date()} 至 {end_date.date()} 沒有觀測資料")
    print(data.head())
    print("Null values before cleaning:")
    print(data.isnull().sum())
    print("Columns:", list(data.columns))

    data["日期"] = pd.to_datetime(data["日期"])
    data = data.sort_values("日期").reset_index(drop=True)
    print("Timestamps sorted:", data["日期"].is_monotonic_increasing)

    if "空氣品質指標" in data.columns:
        print("Air quality indicator values:", set(data["空氣品質指標"]))
    data = data.drop(columns=["空氣品質指標", "測站"], errors="ignore")

    missing_marker_rows = data[(data == "-").any(axis=1)]
    print("Rows containing '-':", len(missing_marker_rows))

    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data[FEATURES] = data[FEATURES].interpolate(method="linear")

    expected = pd.date_range(
        start=data["日期"].min(),
        end=data["日期"].max(),
        freq="h",
    )
    missing_times = expected.difference(data["日期"])
    print("Expected timestamps:", len(expected))
    print("Actual rows:", len(data))
    print("Missing timestamps:", len(missing_times))
    print(missing_times)
    print("Time differences:")
    print(data["日期"].diff().value_counts())

    return data, FEATURES.copy()
