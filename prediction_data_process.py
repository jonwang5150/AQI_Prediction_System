"""將現成爬蟲資料轉為兩種預測模型共用的觀測欄位。"""

from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np


CRAWLER_COLUMNS = {
    "station": "測站", "observed_at": "日期", "aqi": "AQI",
    "o3": "O3", "pm25": "PM2.5", "pm10": "PM10",
    "co": "CO", "so2": "SO2", "no2": "NO2",
}


def validate_prediction_period(start, end):
    """預測區間一律使用台灣當地無時區整點，採 [start, end)。"""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if (pd.isna(start) or pd.isna(end) or start.tzinfo is not None or end.tzinfo is not None
            or start != start.floor("h") or end != end.floor("h") or start >= end):
        raise ValueError("start/end 必須是無時區的整點時間，且 start < end")
    return start, end


def read_database_observations(connection, station, start, end, *, model_columns=True):
    """查詢包含兩端的逐時觀測；模型用中文欄位，更新合併用原始英文欄位。"""
    from sqlalchemy import select
    from db_models import AirQualityHourly
    columns = [getattr(AirQualityHourly, name).label(label if model_columns else name)
               for name, label in CRAWLER_COLUMNS.items() if not model_columns or name != "station"]
    statement = select(*columns).where(
        AirQualityHourly.station == station,
        AirQualityHourly.observed_at >= pd.Timestamp(start).to_pydatetime(),
        AirQualityHourly.observed_at <= pd.Timestamp(end).to_pydatetime(),
    ).order_by(AirQualityHourly.observed_at)
    return pd.read_sql(statement, connection)


def resolve_observation_data(data, paths, database_loader):
    """只在未提供資料及檔案時才呼叫 DB；空資料不可觸發 fallback。"""
    if data is not None and paths is not None:
        raise ValueError("data 與 --data 不可同時提供")
    if data is not None:
        return data
    if paths is not None:
        return read_observation_files(paths)
    return database_loader()


def complete_observation_timeline(data, start, end, *, require_bounds=False):
    """補齊包含兩端的小時列，不補數值；多步回測可要求首尾已取得。"""
    data = data.copy()
    if data.empty:
        raise ValueError("指定測站與期間沒有觀測資料")
    data["日期"] = pd.to_datetime(data["日期"])
    if data["日期"].duplicated().any():
        raise ValueError("觀測時間重複")
    if data["日期"].isna().any() or not data["日期"].eq(data["日期"].dt.floor("h")).all():
        raise ValueError("觀測資料含無效或非整點時間")
    if require_bounds and (data["日期"].min() > start or data["日期"].max() < end):
        raise ValueError(f"觀測範圍不足，需要 {start} 至 {end}，不會外推補齊首尾")
    timeline = pd.date_range(start, end, freq="h")
    missing = len(timeline.difference(pd.DatetimeIndex(data["日期"])))
    frame = data.set_index("日期").reindex(timeline).rename_axis("日期").reset_index()
    return frame, missing


def scale_observation_features(data, features, mean, scale):
    """只以前向補值準備輸入，不修改原始觀測或評估真實值。"""
    missing = set(features).difference(data.columns)
    if missing:
        raise ValueError(f"缺少模型特徵：{sorted(missing)}")
    values = data[features].apply(pd.to_numeric, errors="coerce")
    missing_values = int(values.isna().sum().sum())
    array = values.ffill().to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("特徵前向補值後仍含空值或無限值，請補齊原始觀測")
    return (array - mean) / scale, missing_values


def prepare_observation_data(
    data: pd.DataFrame | Iterable[dict], station: str,
    query_start: pd.Timestamp, input_end: pd.Timestamp,
) -> pd.DataFrame:
    """接受 DataFrame 或 parse_download_rows() 結果，不修改呼叫端資料。"""
    frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if frame.empty:
        raise ValueError("提供的觀測資料為空，不會改查資料庫")
    aliases = {key: value for key, value in CRAWLER_COLUMNS.items() if key in frame.columns}
    if any(value in frame.columns for value in aliases.values()):
        raise ValueError("觀測資料同時包含中英文對應欄位，無法判定使用哪一欄")
    frame = frame.rename(columns=aliases)
    if "日期" not in frame.columns:
        raise ValueError("觀測資料缺少日期／observed_at 欄位")
    if "測站" in frame.columns:
        frame = frame.loc[frame["測站"].astype(str).str.strip() == station.strip()].copy()
    # 無測站欄位代表呼叫端已挑選好指定測站。
    frame["日期"] = pd.to_datetime(frame["日期"], errors="raise")
    if frame["日期"].isna().any() or frame["日期"].dt.tz is not None:
        raise ValueError("觀測日期必須為有效的台灣當地時間（無時區）")
    frame = frame.loc[frame["日期"].between(query_start, input_end)].copy()
    if frame.empty:
        raise ValueError(f"提供的資料沒有測站「{station}」在指定期間的觀測，不會改查資料庫")
    if not frame["日期"].eq(frame["日期"].dt.floor("h")).all():
        raise ValueError("觀測資料含非整點時間")
    if frame["日期"].duplicated().any():
        raise ValueError("觀測時間重複，請先整理重疊的下載批次")
    for column in set(CRAWLER_COLUMNS.values()) - {"測站", "日期"}:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("日期").reset_index(drop=True)


def read_observation_files(paths: Iterable[Path]) -> pd.DataFrame:
    """合併下載批次；支援 CSV、ODS、XLSX、XLS，不自動猜測檔案位置。"""
    frames = []
    for path in paths:
        path = Path(path)
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, encoding="utf-8-sig")
        elif path.suffix.lower() in {".ods", ".xlsx", ".xls"}:
            frame = pd.read_excel(path)
        else:
            raise ValueError(f"不支援的觀測資料格式：{path.suffix}")
        frame.columns = frame.columns.astype(str).str.strip()
        frames.append(frame)
    if not frames:
        raise ValueError("請提供至少一個觀測資料檔")
    return pd.concat(frames, ignore_index=True)
