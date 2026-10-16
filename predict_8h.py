#!/usr/bin/env python3
"""使用現成爬蟲資料或 PostgreSQL 逐時回測 8h 模型，目標區間為 [start, end)。

預設：python predict_8h.py
包含 8/1 全天：python predict_8h.py --end 2026-08-02
僅檢查資料與推論：python predict_8h.py --dry-run
現成檔案：python predict_8h.py --data downloaded.csv --dry-run
記憶體資料：main(data=crawled_rows, argv=["--station", "木柵站"])
未提供資料時才查詢 PostgreSQL。

target_at 使用資料來源的台灣當地時間（無時區）；created_at 使用帶時區時間。
每個目標小時儲存 h1～h8，基準時間可由 target_at - horizon_hours 推得。
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from config import settings
from db_models import AirQualityPrediction8h, upsert_prediction_8h_batch
from pipeline_v2.model_v2 import MultiHorizonLSTM
from update_db import create_database_engine
from model_checkpoint import load_checkpoint_data, preprocessing_from_checkpoint, restore_model
from prediction_data_process import (
    prepare_observation_data, read_database_observations, resolve_observation_data,
    validate_prediction_period, complete_observation_timeline, scale_observation_features,
)


PROJECT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="木柵站")
    parser.add_argument("--data", type=Path, nargs="+", help="現成爬蟲資料檔（CSV／ODS／XLSX／XLS），可指定多個；略過資料庫讀取")
    parser.add_argument("--model", type=Path, default=PROJECT_DIR / "aqi_lstm_8h_model.pth")
    parser.add_argument("--start", default="2026-01-01", help="目標起始時間（包含）")
    parser.add_argument("--end", default="2026-08-01", help="目標結束時間（不包含）")
    parser.add_argument("--batch-size", type=int, default=256, help="推論批次大小")
    parser.add_argument("--dry-run", action="store_true", help="讀取與推論，但不建表或寫入")
    parser.add_argument("--latest-origin", help="最新觀測整點；只用此前資料，允許預測其後 8 小時")
    return parser


def load_checkpoint(path: Path):
    """只載入可信任的本機 checkpoint，沿用訓練時的 scaler。"""
    model_version = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
    checkpoint = load_checkpoint_data(path, required=("forecast_horizon",))
    features, window_size, mean, scale = preprocessing_from_checkpoint(checkpoint)
    horizon = int(checkpoint["forecast_horizon"])
    if horizon != 8 or checkpoint["model_config"].get("forecast_horizon") != 8:
        raise ValueError("此程式需要 forecast_horizon=8 的模型")
    model = restore_model(MultiHorizonLSTM, checkpoint)
    return model, features, window_size, horizon, mean, scale, model_version


def load_database_data(engine, station, query_start, input_end):
    with engine.connect() as connection:
        return read_database_observations(connection, station, query_start, input_end)


def prepare_inputs(data, features, query_start, input_end, mean, scale):
    """僅向前補值；拒絕將整段尚未取得的觀測延伸成回測資料。"""
    frame, missing_hours = complete_observation_timeline(
        data, query_start, input_end, require_bounds=True)
    inputs, missing_values = scale_observation_features(frame, features, mean, scale)
    return inputs, missing_hours, missing_values


def predict_period(model, inputs, window_size, horizon, start, end, batch_size=256, latest_origin=None):
    """每個基準時間推論一次，再裁切至目標區間；每個目標有完整 h1～h8。"""
    if batch_size <= 0 or window_size <= 0:
        raise ValueError("batch_size 與 window_size 必須大於 0")
    last_origin = end - pd.Timedelta(hours=2) if latest_origin is None else pd.Timestamp(latest_origin)
    origins = pd.date_range(start - pd.Timedelta(hours=horizon), last_origin, freq="h")
    if len(inputs) != len(origins) + window_size - 1:
        raise ValueError("輸入資料長度與預測期間不一致")
    rows = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(origins), batch_size):
            stop = min(offset + batch_size, len(origins))
            # windows.shape==(256,24,特徵數)
            windows = np.stack([inputs[i:i + window_size] for i in range(offset, stop)])
            predictions = model(torch.from_numpy(windows.astype(np.float32))).cpu().numpy()
            if predictions.shape != (stop - offset, horizon) or not np.isfinite(predictions).all():
                raise RuntimeError("模型輸出維度錯誤或含非有限值")
            for origin, vector in zip(origins[offset:stop], predictions):
                for h, value in enumerate(vector, start=1):
                    target = origin + pd.Timedelta(hours=h)
                    if start <= target < end:
                        rows.append({"target_at": target.to_pydatetime(),
                                     "horizon_hours": h, "predicted_aqi": float(value)})
    result = pd.DataFrame(rows).sort_values(["target_at", "horizon_hours"]).reset_index(drop=True)
    targets = pd.date_range(start, end, freq="h", inclusive="left")
    expected = sum(int(((targets - pd.Timedelta(hours=h)) <= last_origin).sum())
                   for h in range(1, horizon + 1))
    if len(result) != expected or result.duplicated(["target_at", "horizon_hours"]).any():
        raise RuntimeError("預測筆數或唯一鍵不符合預期")
    return result


def main(data=None, argv: list[str] | None = None) -> int:
    """data 可為 DataFrame 或爬蟲列的 iterable；提供資料且 dry-run 時不建立 DB engine。"""
    args = build_parser().parse_args(argv)
    if data is not None and args.data is not None:
        raise ValueError("data 與 --data 不可同時提供")
    start, end = validate_prediction_period(args.start, args.end)
    if not args.station.strip() or args.batch_size <= 0:
        raise ValueError("測站不可空白，batch-size 必須大於 0")
    model, features, window, horizon, mean, scale, version = load_checkpoint(args.model)
    # 最早目標的 h8：基準 start-8h；24h 窗口從 start-31h 開始。
    query_start = start - pd.Timedelta(hours=horizon + window - 1)
    input_end = end - pd.Timedelta(hours=2)
    latest_origin = pd.Timestamp(args.latest_origin) if args.latest_origin else None
    if latest_origin is not None:
        if (pd.isna(latest_origin) or latest_origin.tzinfo is not None
                or latest_origin != latest_origin.floor("h") or latest_origin < start - pd.Timedelta(hours=1)
                or end != latest_origin + pd.Timedelta(hours=horizon + 1)):
            raise ValueError("latest-origin 必須為不早於 start 前一小時的整點，end 須為 latest-origin + 9 小時")
        input_end = latest_origin
    engine = None

    def load_from_database():
        nonlocal engine
        engine = create_database_engine(settings)
        return load_database_data(engine, args.station.strip(), query_start, input_end)

    try:
        raw = resolve_observation_data(data, args.data, load_from_database)
        raw = prepare_observation_data(raw, args.station, query_start, input_end)
        inputs, missing_hours, missing_values = prepare_inputs(
            raw, features, query_start, input_end, mean, scale)
        result = predict_period(model, inputs, window, horizon, start, end, args.batch_size,
                                latest_origin=latest_origin)
        print(f"測站：{args.station}；目標區間：[{start}, {end})")
        print(f"模型版本：{version}")
        print(f"缺少整點：{missing_hours}；前向補值：{missing_values} 格")
        print(f"產生 {len(result)} 筆預測" + ("，含最新基準的未來 8 小時" if latest_origin is not None else "，每個目標時間包含 h1～h8"))
        print(result.head(8).to_string(index=False))
        if not args.dry_run:
            if engine is None:
                engine = create_database_engine(settings)
            created_at = datetime.now(timezone.utc)
            rows = [{**row, "station": args.station.strip(), "model_version": version,
                     "created_at": created_at} for row in result.to_dict(orient="records")]
            # 建表與全部批次寫入使用同一交易；失敗時全部回復。
            with engine.begin() as connection:
                AirQualityPrediction8h.__table__.create(connection, checkfirst=True)
                with Session(bind=connection) as session:
                    upsert_prediction_8h_batch(session, rows)
            print("已寫入 public.air_quality_predictions_8h")
        else:
            print("dry-run：未建表或寫入資料")
    finally:
        if engine is not None:
            engine.dispose()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SQLAlchemyError as exc:
        print(f"資料庫操作失敗（{type(exc).__name__}），請檢查連線、資料表及權限。", file=sys.stderr)
        raise SystemExit(1)
    except (OSError, ValueError, RuntimeError, pickle.UnpicklingError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        raise SystemExit(1)
