"""Chronological split, normalization, and sequence construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# 封裝單步模型的訓練、測試資料與前處理設定。
@dataclass
class DatasetBundle:
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


# 以連續小時視窗建立特徵序列及下一小時 AQI 標籤。
def _make_sequences(
    data: pd.DataFrame,
    features: list[str],
    window_size: int,
    target_start: pd.Timestamp | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    targets: list[float] = []

    for index in range(window_size, len(data)):
        window = data.iloc[index - window_size:index]
        if not (window["日期"].diff().dropna() == pd.Timedelta(hours=1)).all():
            continue
        if data.iloc[index]["日期"] - window["日期"].iloc[-1] != pd.Timedelta(hours=1):
            continue
        if target_start is not None and data.iloc[index]["日期"] < target_start:
            continue
        sequences.append(window[features].to_numpy())
        targets.append(float(data.iloc[index]["AQI"]))

    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def prepare_datasets(
    data: pd.DataFrame,
    features: list[str],
    window_size: int = 24,
    test_size: float = 0.2,
    batch_size: int = 32,
) -> DatasetBundle:
    """Split chronologically, fit the scaler on train data, and build loaders."""
    split_index = int(len(data) * (1 - test_size))
    train_df = data.iloc[:split_index].copy()
    test_df = data.iloc[split_index:].copy()

    print("Training period:", train_df["日期"].min(), "~", train_df["日期"].max())
    print("Testing period:", test_df["日期"].min(), "~", test_df["日期"].max())

    scaler = StandardScaler()
    train_df.loc[:, features] = scaler.fit_transform(train_df[features])
    test_df.loc[:, features] = scaler.transform(test_df[features])

    train_X, train_y = _make_sequences(train_df, features, window_size)

    history = train_df.tail(window_size)
    test_data = pd.concat([history, test_df], ignore_index=True)
    test_X, test_y = _make_sequences(
        test_data,
        features,
        window_size,
        target_start=test_df["日期"].min(),
    )

    print("train_X:", train_X.shape)
    print("train_y:", train_y.shape)
    print("test_X:", test_X.shape)
    print("test_y:", test_y.shape)

    train_dataset = TensorDataset(
        torch.tensor(train_X, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(test_X, dtype=torch.float32),
        torch.tensor(test_y, dtype=torch.float32),
    )

    return DatasetBundle(
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
    )
