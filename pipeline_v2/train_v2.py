"""多步 AQI 模型的訓練與逐預測距離評估。"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from .model_v2 import MultiHorizonLSTM
except ImportError:  # 支援直接執行 pipeline_v2 內的程式
    from model_v2 import MultiHorizonLSTM


def train_model_new(
    model: MultiHorizonLSTM,
    train_loader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 0.001,
) -> MultiHorizonLSTM:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for inputs, targets in train_loader:
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / total_samples:.4f}")
    return model


def evaluate_model_new(
    model: MultiHorizonLSTM,
    test_loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    prediction_batches = []
    target_batches = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            prediction_batches.append(model(inputs))
            target_batches.append(targets)

    predictions = torch.cat(prediction_batches)
    targets = torch.cat(target_batches)
    mae = torch.mean(torch.abs(predictions - targets), dim=0)
    rmse = torch.sqrt(torch.mean((predictions - targets) ** 2, dim=0))

    for horizon, (horizon_mae, horizon_rmse) in enumerate(zip(mae, rmse), start=1):
        print(
            f"提前 {horizon} 小時："
            f"MAE={horizon_mae.item():.4f}, RMSE={horizon_rmse.item():.4f}"
        )
    print(f"整體：MAE={mae.mean().item():.4f}, RMSE={rmse.mean().item():.4f}")
    return mae.cpu().numpy(), rmse.cpu().numpy()
