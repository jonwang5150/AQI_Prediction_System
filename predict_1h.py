#!/usr/bin/env python3
"""使用現成爬蟲資料或 PostgreSQL 觀測，以既有 checkpoint 預測逐時 AQI。

現成檔案：python predict_1h.py --data downloaded.csv
記憶體資料：main(data=crawled_rows, argv=["--station", "木柵站"])
未提供資料時才查詢 PostgreSQL；預測結果仍寫入資料庫。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sqlalchemy.exc import SQLAlchemyError

import update_db
from config import settings
from pipeline_v1.model import LSTMModel
from model_checkpoint import load_checkpoint_data, preprocessing_from_checkpoint, restore_model
from prediction_data_process import (
    prepare_observation_data, read_database_observations, resolve_observation_data,
    validate_prediction_period, complete_observation_timeline, scale_observation_features,
)



PROJECT_DIR = Path(__file__).resolve().parent
PREDICTION_START = pd.Timestamp("2026-01-01 00:00:00")
PREDICTION_END = pd.Timestamp("2026-08-01 00:00:00")


# 定義單步推論的測站、目標期間、輸入檔案與模型參數。
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


# 還原單步模型及訓練時的特徵順序、視窗與縮放參數。
def load_checkpoint(path: Path) -> tuple[LSTMModel, list[str], int, np.ndarray, np.ndarray]:
    checkpoint = load_checkpoint_data(path)
    features, window_size, scaler_mean, scaler_scale = preprocessing_from_checkpoint(checkpoint)
    model = restore_model(LSTMModel, checkpoint)
    return model, features, window_size, scaler_mean, scaler_scale


# 讀取指定測站的觀測資料，範圍不包含預測區間終點。
def load_database_data(station: str, query_start: pd.Timestamp, end=None) -> pd.DataFrame:
    end = PREDICTION_END if end is None else pd.Timestamp(end)
    with update_db.database_connection(settings) as connection:
        return read_database_observations(
            connection, station, query_start, end - pd.Timedelta(hours=1))


def complete_hourly_timeline(
    data: pd.DataFrame,
    query_start: pd.Timestamp,
    end=None,
) -> tuple[pd.DataFrame, int]:
    """補齊缺少的整點列，並回傳補入的時間列數。"""
    end = PREDICTION_END if end is None else pd.Timestamp(end)
    return complete_observation_timeline(data, query_start, end - pd.Timedelta(hours=1))


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

    start_time, end_time = validate_prediction_period(start, end)

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
    start, end = validate_prediction_period(start, end)
    model, features, window_size, scaler_mean, scaler_scale = load_checkpoint(args.model)
    query_start = start - pd.Timedelta(hours=window_size)

    data = resolve_observation_data(
        data, args.data, lambda: load_database_data(args.station, query_start, end))
    data = prepare_observation_data(
        data, args.station, query_start, end - pd.Timedelta(hours=1))
    data, missing_hours = complete_hourly_timeline(data, query_start, end)
    missing_features = {"AQI", *features}.difference(data.columns)
    if missing_features:
        raise ValueError(f"觀測資料缺少必要欄位：{', '.join(sorted(missing_features))}")
    # 真實值保留原始缺口，不能把前向補值當成評估用觀測。
    period_actual = data.loc[data["日期"].between(start, end, inclusive="left"), ["日期", "AQI"]].copy()
    values, missing_before = scale_observation_features(data, features, scaler_mean, scaler_scale)
    scaled_data = data.copy()
    scaled_data.loc[:, features] = values
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
