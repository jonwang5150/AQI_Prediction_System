"""訓練並評估一次預測未來 8 小時 AQI 的新模型。"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .data_cleaning_v2 import load_and_clean_data_new
    from .data_split_v2 import prepare_datasets_new
    from .model_v2 import MultiHorizonLSTM, save_checkpoint_new
    from .predict_v2 import predict_next_hours
    from .train_v2 import evaluate_model_new, train_model_new
except ImportError:  # 支援 python pipeline_v2/run_model_v2.py
    from data_cleaning_v2 import load_and_clean_data_new
    from data_split_v2 import prepare_datasets_new
    from model_v2 import MultiHorizonLSTM, save_checkpoint_new
    from predict_v2 import predict_next_hours
    from train_v2 import evaluate_model_new, train_model_new


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="木柵站", help="資料庫測站名稱")
    parser.add_argument("--start", default="2025-01-01", help="訓練資料開始日期（含）")
    parser.add_argument("--end", default="2025-12-31", help="訓練資料結束日期（含全天）")
    parser.add_argument("--model", type=Path, default=Path("aqi_lstm_8h_model.pth"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--forecast-horizon", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data, features = load_and_clean_data_new(station=args.station, start=args.start, end=args.end)
    datasets = prepare_datasets_new(
        data,
        features,
        window_size=args.window_size,
        forecast_horizon=args.forecast_horizon,
    )

    model = MultiHorizonLSTM(
        input_size=len(features),
        forecast_horizon=args.forecast_horizon,
    )
    train_model_new(model, datasets.train_loader, epochs=args.epochs)
    evaluate_model_new(model, datasets.test_loader)

    future = predict_next_hours(
        model,
        datasets.test_df,
        data,
        datasets.features,
        datasets.window_size,
        datasets.forecast_horizon,
    )
    print("最新窗口的未來預測：")
    print(future.to_string(index=False))

    save_checkpoint_new(
        model,
        args.model,
        datasets.features,
        datasets.window_size,
        datasets.forecast_horizon,
        datasets.scaler,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
