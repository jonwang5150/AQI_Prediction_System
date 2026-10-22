"""Run the complete AQI prediction pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .data_cleaning import load_and_clean_data
    from .data_split import prepare_datasets
    from .model import LSTMModel, save_checkpoint
    from .predict import predict_next_hour
    from .train import evaluate_model, train_model
    from .visualization import show_visualizations
except ImportError:  # 支援 python pipeline_v1/run_model.py
    from data_cleaning import load_and_clean_data
    from data_split import prepare_datasets
    from model import LSTMModel, save_checkpoint
    from predict import predict_next_hour
    from train import evaluate_model, train_model
    from visualization import show_visualizations


# 定義單步模型訓練的資料期間、模型路徑與訓練參數。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AQI LSTM prediction pipeline")
    parser.add_argument("--station", default="木柵站", help="資料庫測站名稱")
    parser.add_argument("--start", default="2025-01-01", help="訓練資料開始日期（含）")
    parser.add_argument("--end", default="2025-12-31", help="訓練資料結束日期（含全天）")
    parser.add_argument("--model", type=Path, default=Path("aqi_lstm_model.pth"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--no-visualization", action="store_true")
    return parser


# 串接資料清理、訓練、評估與預測，最後保存單步模型。
def main() -> int:
    args = build_parser().parse_args()
    data, features = load_and_clean_data(station=args.station, start=args.start, end=args.end)

    if not args.no_visualization:
        show_visualizations(data)

    datasets = prepare_datasets(
        data,
        features,
        window_size=args.window_size,
    )
    model = LSTMModel(input_size=len(features))
    train_model(model, datasets.train_loader, epochs=args.epochs)
    evaluate_model(model, datasets.test_loader)
    predict_next_hour(
        model,
        datasets.test_df,
        data,
        datasets.features,
        datasets.window_size,
    )
    save_checkpoint(
        model,
        args.model,
        datasets.features,
        datasets.window_size,
        datasets.scaler,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
