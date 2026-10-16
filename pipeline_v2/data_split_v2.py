"""多步 AQI 預測的時間切分、標準化與序列建立。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


TARGET_COLUMN = "_target_aqi"


@dataclass
class MultiHorizonDatasetBundle:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_loader: DataLoader
    test_loader: DataLoader
    train_X: np.ndarray
    train_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    scaler: StandardScaler
    features: list[str]
    window_size: int
    forecast_horizon: int


def _make_sequences_new(
    data: pd.DataFrame,
    features: list[str],
    window_size: int,
    forecast_horizon: int,
    target_start: pd.Timestamp | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    last_start = len(data) - forecast_horizon + 1

    for index in range(window_size, last_start):
        window = data.iloc[index - window_size:index]
        future = data.iloc[index:index + forecast_horizon]
        complete_span = data.iloc[index - window_size:index + forecast_horizon]

        if target_start is not None and future["日期"].iloc[0] < target_start:
            continue
        if not (
            complete_span["日期"].diff().dropna() == pd.Timedelta(hours=1)
        ).all():
            continue
        if future[TARGET_COLUMN].isna().any():
            continue

        sequences.append(window[features].to_numpy())
        targets.append(future[TARGET_COLUMN].to_numpy())

    if not sequences:
        return (
            np.empty((0, window_size, len(features)), dtype=np.float32),
            np.empty((0, forecast_horizon), dtype=np.float32),
        )
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def prepare_datasets_new(
    data: pd.DataFrame,
    features: list[str],
    window_size: int = 24,
    forecast_horizon: int = 8,
    test_size: float = 0.2,
    batch_size: int = 32,
) -> MultiHorizonDatasetBundle:
    """依時間切分資料，以訓練集縮放，建立 (N, 24, 7) → (N, 8)。"""
    if window_size <= 0 or forecast_horizon <= 0:
        raise ValueError("window_size 與 forecast_horizon 必須大於 0")
    if not 0 < test_size < 1:
        raise ValueError("test_size 必須介於 0 和 1 之間")

    split_index = int(len(data) * (1 - test_size))
    train_df = data.iloc[:split_index].copy()
    test_df = data.iloc[split_index:].copy()
    if len(train_df) < window_size + forecast_horizon or len(test_df) < forecast_horizon:
        raise ValueError("資料量不足以建立訓練與測試序列")

    # AQI 同時是輸入與標籤：輸入需要標準化，標籤保留原始 AQI 尺度。
    train_df[TARGET_COLUMN] = train_df["AQI"].astype(float)
    test_df[TARGET_COLUMN] = test_df["AQI"].astype(float)
    scaler = StandardScaler()
    train_df.loc[:, features] = scaler.fit_transform(train_df[features])
    test_df.loc[:, features] = scaler.transform(test_df[features])

    train_X, train_y = _make_sequences_new(
        train_df,
        features,
        window_size,
        forecast_horizon,
    )
    history = train_df.tail(window_size)
    test_data = pd.concat([history, test_df], ignore_index=True)
    test_X, test_y = _make_sequences_new(
        test_data,
        features,
        window_size,
        forecast_horizon,
        target_start=test_df["日期"].min(),
    )
    if len(train_X) == 0 or len(test_X) == 0:
        raise ValueError("時間缺口造成無法建立訓練或測試序列")

    print(f"訓練期間：{train_df['日期'].min()} ～ {train_df['日期'].max()}")
    print(f"測試期間：{test_df['日期'].min()} ～ {test_df['日期'].max()}")
    print(f"train_X: {train_X.shape}, train_y: {train_y.shape}")
    print(f"test_X: {test_X.shape}, test_y: {test_y.shape}")

    train_dataset = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
    test_dataset = TensorDataset(torch.from_numpy(test_X), torch.from_numpy(test_y))
    return MultiHorizonDatasetBundle(
        train_df=train_df,
        test_df=test_df,
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        train_X=train_X,
        train_y=train_y,
        test_X=test_X,
        test_y=test_y,
        scaler=scaler,
        features=features,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
    )
