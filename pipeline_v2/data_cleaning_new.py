"""多步 AQI 預測使用的資料清理。"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import create_engine

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings
from prediction_data_process import read_database_observations


FEATURES_NEW = ["AQI", "O3", "PM2.5", "PM10", "CO", "SO2", "NO2"]


def load_and_clean_data_new(
    *,
    station: str = "木柵站",
    start: str = "2025-01-01",
    end: str = "2025-12-31",
) -> tuple[pd.DataFrame, list[str]]:
    """從 DB 載入逐時資料並清理；日期範圍包含結束日全天。"""
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    if (pd.isna(start_date) or pd.isna(end_date)
            or start_date.tzinfo is not None or end_date.tzinfo is not None
            or start_date != start_date.normalize() or end_date != end_date.normalize()
            or start_date > end_date):
        raise ValueError("start/end 必須是有效的無時區日期，且 start 不晚於 end")
    station = station.strip()
    if not station:
        raise ValueError("station 不可為空白")
    end_time = end_date + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            data = read_database_observations(connection, station, start_date, end_time)
    finally:
        engine.dispose()
    if data.empty:
        raise ValueError(f"測站「{station}」在 {start_date.date()} 至 {end_date.date()} 沒有觀測資料")
    required = {"日期", *FEATURES_NEW}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"資料缺少欄位：{', '.join(sorted(missing))}")

    data["日期"] = pd.to_datetime(data["日期"], errors="raise")
    data = data.sort_values("日期").reset_index(drop=True)
    if data["日期"].duplicated().any():
        raise ValueError("資料含有重複的日期時間")

    data = data.drop(columns=["空氣品質指標", "測站"], errors="ignore")
    for column in FEATURES_NEW:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data[FEATURES_NEW] = data[FEATURES_NEW].astype(float)

    missing_before = int(data[FEATURES_NEW].isna().sum().sum())
    # 線性插值會利用缺口前後的觀測值，適合歷史資料訓練與回測。
    data.loc[:, FEATURES_NEW] = data[FEATURES_NEW].interpolate(method="linear")
    if data[FEATURES_NEW].isna().any().any():
        raise ValueError("線性插值後仍有空值；資料開頭必須具有完整觀測")

    expected = pd.date_range(data["日期"].min(), data["日期"].max(), freq="h")
    missing_times = expected.difference(data["日期"])
    print(f"資料期間：{data['日期'].min()} ～ {data['日期'].max()}")
    print(f"線性補值：{missing_before} 格；缺少整點：{len(missing_times)} 筆")
    return data, FEATURES_NEW.copy()
