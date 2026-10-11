"""將現成爬蟲資料轉為兩種預測模型共用的觀測欄位。"""

from pathlib import Path
from typing import Iterable

import pandas as pd


CRAWLER_COLUMNS = {
    "station": "測站", "observed_at": "日期", "aqi": "AQI",
    "o3": "O3", "pm25": "PM2.5", "pm10": "PM10",
    "co": "CO", "so2": "SO2", "no2": "NO2",
}


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
