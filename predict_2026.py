#!/usr/bin/env python3
"""使用現成爬蟲資料或 PostgreSQL 觀測，以既有 checkpoint 預測逐時 AQI。

現成檔案：python predict_2026.py --data downloaded.csv
記憶體資料：main(data=crawled_rows, argv=["--station", "木柵站"])
未提供資料時才查詢 PostgreSQL；預測結果仍寫入資料庫。
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine, select
from sqlalchemy.exc import SQLAlchemyError

import update_db
from config import settings
from db_models import AirQualityHourly
from pipeline_v1.model import LSTMModel
from prediction_data_process import prepare_observation_data, read_observation_files



PROJECT_DIR = Path(__file__).resolve().parent
PREDICTION_START = pd.Timestamp("2026-01-01 00:00:00")
PREDICTION_END = pd.Timestamp("2026-08-01 00:00:00")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="木柵站", help="資料庫中的測站名稱")
    parser.add_argument("--start", default=None, help="目標開始時間（包含），預設 2026-01-01")
    parser.add_argument("--end", default=None, help="目標結束時間（不包含），預設 2026-08-01")
    parser.add_argument("--data", type=Path, nargs="+", help="現成爬蟲資料檔（CSV／ODS／XLSX／XLS），可指定多個；略過資料庫讀取")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_DIR / "aqi_lstm_model.pth",
        help="模型 checkpoint 路徑",
    )
    return parser


def load_checkpoint(path: Path) -> tuple[LSTMModel, list[str], int, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到模型檔：{path}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # 相容尚未提供 weights_only 參數的舊版 PyTorch
        checkpoint = torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        # 專案既有 checkpoint 含 NumPy array；PyTorch 2.6 的安全模式不接受它。
        # 請只對可信任的本機模型使用此 fallback。
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    required = {
        "model_state_dict",
        "model_config",
        "features",
        "window_size",
        "scaler_mean",
        "scaler_scale",
    }
    # difference()：找出「存在於 required，但不存在於 checkpoint」的項目。
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"模型 checkpoint 缺少欄位：{', '.join(sorted(missing))}")

    features = list(checkpoint["features"])
    window_size = int(checkpoint["window_size"])
    scaler_mean = np.asarray(checkpoint["scaler_mean"], dtype=float)
    scaler_scale = np.asarray(checkpoint["scaler_scale"], dtype=float)
    if len(features) != len(scaler_mean) or len(features) != len(scaler_scale):
        raise ValueError("checkpoint 的 features 與 scaler 維度不一致")
    if np.any(scaler_scale == 0):
        raise ValueError("checkpoint 含有無效的 scaler_scale=0")

    model = LSTMModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, features, window_size, scaler_mean, scaler_scale


def load_database_data(station: str, query_start: pd.Timestamp, end=None) -> pd.DataFrame:
    end = PREDICTION_END if end is None else pd.Timestamp(end)
    columns = {
        "O3": AirQualityHourly.o3,
        "PM2.5": AirQualityHourly.pm25,
        "PM10": AirQualityHourly.pm10,
        "CO": AirQualityHourly.co,
        "SO2": AirQualityHourly.so2,
        "NO2": AirQualityHourly.no2,
    }
    statement = (
        select(
            AirQualityHourly.observed_at.label("日期"),
            AirQualityHourly.aqi.label("AQI"),
            *[column.label(name) for name, column in columns.items()],
        )
        .where(
            AirQualityHourly.station == station,
            AirQualityHourly.observed_at >= query_start.to_pydatetime(),
            AirQualityHourly.observed_at < end.to_pydatetime(),
        )
        .order_by(AirQualityHourly.observed_at)
    )

    engine = create_engine(
        update_db.database_url_from_settings(settings),
        connect_args={"connect_timeout": 10},
        hide_parameters=True,
    )
    try:
        with engine.connect() as connection:
            return pd.read_sql(statement, connection)
    finally:
        engine.dispose()


def complete_hourly_timeline(
    data: pd.DataFrame,
    query_start: pd.Timestamp,
    end=None,
) -> tuple[pd.DataFrame, int]:
    """補齊缺少的整點列，並回傳補入的時間列數。"""
    if data.empty:
        raise ValueError("指定的測站與時段在資料庫中沒有資料")
    if data["日期"].duplicated().any():
        duplicates = data.loc[data["日期"].duplicated(), "日期"].dt.strftime("%Y-%m-%d %H:%M")
        raise ValueError(f"資料庫有重複時間：{', '.join(duplicates.head(5))}")

    expected = pd.date_range(query_start, PREDICTION_END if end is None else end, freq="h", inclusive="left")
    actual = pd.DatetimeIndex(data["日期"])
    missing = expected.difference(actual)
    completed = (
        data.set_index("日期")
        .reindex(expected)
        .rename_axis("日期")
        .reset_index()
    )
    return completed, len(missing)

def predict_period(
    model: LSTMModel,
    scaled_data: pd.DataFrame,
    features: list[str],
    window_size: int,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """預測半開區間 ``[start, end)`` 中每個小時的 AQI。"""
    required_columns = {"日期", *features}
    missing_columns = required_columns.difference(scaled_data.columns)
    if missing_columns:
        raise ValueError(f"預測資料缺少欄位：{', '.join(sorted(missing_columns))}")
    if window_size <= 0:
        raise ValueError("window_size 必須大於 0")

    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    if start_time >= end_time:
        raise ValueError("預測開始時間必須早於結束時間")

    data = scaled_data.sort_values("日期").reset_index(drop=True)
    dates = pd.to_datetime(data["日期"])
    sequences = []
    prediction_dates = []

    for index in range(window_size, len(data)):
        target_time = dates.iloc[index]
        if target_time < start_time:
            continue
        if target_time >= end_time:
            break

        window_dates = dates.iloc[index - window_size:index]
        expected_dates = pd.date_range(
            end=target_time - pd.Timedelta(hours=1),
            periods=window_size,
            freq="h",
        )
        if not window_dates.reset_index(drop=True).equals(pd.Series(expected_dates)):
            continue
        sequences.append(data.iloc[index - window_size:index][features].to_numpy())
        prediction_dates.append(target_time)

    if not sequences:
        return pd.DataFrame(columns=["日期", "預測AQI"])

    inputs = torch.from_numpy(np.stack(sequences).astype(np.float32, copy=False))
    model.eval()
    with torch.no_grad():
        predictions = model(inputs).detach().cpu().numpy().reshape(-1)

    return pd.DataFrame({"日期": prediction_dates, "預測AQI": predictions})


def main(data=None, argv: list[str] | None = None) -> int:
    """data 可為 DataFrame 或爬蟲列的 iterable；傳入空資料也不回查資料庫。"""
    args = build_parser().parse_args(argv)
    args.station = args.station.strip()
    if not args.station:
        raise ValueError("測站不可空白")
    if data is not None and args.data is not None:
        raise ValueError("data 與 --data 不可同時提供")
    start = pd.Timestamp(args.start) if args.start else PREDICTION_START
    end = pd.Timestamp(args.end) if args.end else PREDICTION_END
    if (pd.isna(start) or pd.isna(end) or start.tzinfo is not None or end.tzinfo is not None
            or start != start.floor("h") or end != end.floor("h") or start >= end):
        raise ValueError("start/end 必須是無時區的整點時間，且 start < end")
    model, features, window_size, scaler_mean, scaler_scale = load_checkpoint(args.model)
    query_start = start - pd.Timedelta(hours=window_size)

    if data is None:
        data = (read_observation_files(args.data) if args.data is not None
                else load_database_data(args.station, query_start, end))
    data = prepare_observation_data(
        data, args.station, query_start, end - pd.Timedelta(hours=1))
    data["日期"] = pd.to_datetime(data["日期"])
    data, missing_hours = complete_hourly_timeline(data, query_start, end)
    # 真實值保留原始缺口，不能把前向補值當成評估用觀測。
    period_actual = data.loc[data["日期"].between(start, end, inclusive="left"), ["日期", "AQI"]].copy()

    missing_features = {"AQI", *features}.difference(data.columns)
    if missing_features:
        raise ValueError(f"觀測資料缺少必要欄位：{', '.join(sorted(missing_features))}")
    for feature in features:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
    # 僅使用過去觀測值進行前向補值，避免補值時使用未來資料。
    missing_before = int(data[features].isna().sum().sum())
    data.loc[:, features] = data[features].ffill()
    
    if data[features].isna().any().any():
        raise ValueError("模型特徵補值後仍含空值，無法建立預測視窗")

 

    scaled_data = data.copy()
    scaled_data.loc[:, features] = (data[features].to_numpy(dtype=float) - scaler_mean) / scaler_scale
    predictions = predict_period(
        model,
        scaled_data,
        features,
        window_size,
        start,
        end,
    )

    result = predictions.merge(period_actual, on="日期", how="left")
    result = result.rename(columns={"AQI": "實際AQI"})
    expected_count = len(pd.date_range(start, end, freq="h", inclusive="left"))
    if len(result) != expected_count:
        raise RuntimeError(f"預期產生 {expected_count} 筆預測，實際只有 {len(result)} 筆")

    prediction_rows = (
        update_db.parse_prediction_row({**raw, "測站": args.station})
        for raw in result.to_dict(orient="records")
    )
    written_count = update_db.update_prediction_rows(
        prediction_rows,
        settings=settings,
    )
    print(f"測站：{args.station}")
    print(f"預測時段：{start} 至 {end}（不含結束時間）")
    print(f"補齊缺少的逐時資料：{missing_hours} 筆")
    print(f"污染物欄位前向補值：{missing_before} 格")
    print(f"完成 {written_count} 筆逐時 AQI 預測，已寫入 public.air_quality_predictions。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ModuleNotFoundError as exc:
        if exc.name in {"psycopg", "psycopg2"}:
            print(
                '錯誤：缺少 PostgreSQL driver，請執行：pip install "psycopg[binary]>=3,<4"',
                file=sys.stderr,
            )
        else:
            print(f"錯誤：缺少 Python 套件 {exc.name}", file=sys.stderr)
        raise SystemExit(1)
    except (OSError, ValueError, RuntimeError, SQLAlchemyError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        raise SystemExit(1)
