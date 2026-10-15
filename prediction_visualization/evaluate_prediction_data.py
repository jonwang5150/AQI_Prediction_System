"""Evaluate prediction data on shared observation times."""

from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_prediction_data(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """以所有模型共同具有有限預測與真實值的時間點評估，不補值。"""
    aligned = {}
    common = None
    for source, data in datasets.items():
        frame = data.set_index("日期")[["預測AQI", "實際AQI"]].apply(
            pd.to_numeric, errors="coerce")
        if frame.index.has_duplicates:
            raise ValueError("評估資料的目標時間不可重複")
        frame = frame.loc[np.isfinite(frame.to_numpy(dtype=float)).all(axis=1)]
        aligned[source] = frame
        common = frame.index if common is None else common.intersection(frame.index)
    rows = []
    for source, frame in aligned.items():
        paired = frame.loc[common]
        errors = (paired["預測AQI"] - paired["實際AQI"]).to_numpy(dtype=float)
        count = len(errors)
        rows.append({
            "來源": source, "有效筆數": count,
            "未納入筆數": len(datasets[source]) - count,
            "MAE": float(np.abs(errors).mean()) if count else np.nan,
            "RMSE": float(np.sqrt(np.mean(errors ** 2))) if count else np.nan,
            "Bias": float(errors.mean()) if count else np.nan,
            "評估開始": common.min() if count else pd.NaT,
            "評估結束": common.max() if count else pd.NaT,
        })
    return pd.DataFrame(rows)


