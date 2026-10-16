"""使用多步模型一次預測未來數小時 AQI。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

try:
    from .model_v2 import MultiHorizonLSTM
except ImportError:  # 支援直接執行 pipeline_v2 內的程式
    from model_v2 import MultiHorizonLSTM


def predict_next_hours(
    model: MultiHorizonLSTM,
    scaled_data: pd.DataFrame,
    raw_data: pd.DataFrame,
    features: list[str],
    window_size: int = 24,
    forecast_horizon: int = 8,
) -> pd.DataFrame:
    """使用最新 24 小時資料，一次輸出未來 8 小時 AQI。"""
    window = scaled_data.sort_values("日期").tail(window_size)
    if len(window) != window_size:
        raise ValueError(f"需要 {window_size} 筆歷史資料，實際只有 {len(window)} 筆")
    if not (window["日期"].diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("最新歷史窗口不是連續逐時資料")

    inputs = torch.from_numpy(
        window[features].to_numpy(dtype=np.float32, copy=True)
    ).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        predictions = model(inputs).cpu().numpy().reshape(-1)
    if len(predictions) != forecast_horizon:
        raise RuntimeError(
            f"模型應輸出 {forecast_horizon} 筆，實際輸出 {len(predictions)} 筆"
        )

    first_time = window["日期"].iloc[-1] + pd.Timedelta(hours=1)
    prediction_dates = pd.date_range(first_time, periods=forecast_horizon, freq="h")
    result = pd.DataFrame({"日期": prediction_dates, "預測AQI": predictions})
    actual = raw_data[["日期", "AQI"]].rename(columns={"AQI": "實際AQI"})
    return result.merge(actual, on="日期", how="left")
