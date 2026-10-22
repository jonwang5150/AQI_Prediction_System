"""Training and evaluation functions."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from .model import LSTMModel
except ImportError:  # 支援直接執行 pipeline_v1 內的程式
    from model import LSTMModel


# 以均方誤差與 Adam 更新模型權重，回傳訓練完成的模型。
def train_model(
    model: LSTMModel,
    train_loader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 0.001,
) -> LSTMModel:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for features, targets in train_loader:
            predictions = model(features)
            loss = criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)

        average_loss = total_loss / total_samples
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")

    return model


# 在測試集上推論，計算並回傳整體 MAE 與 RMSE。
def evaluate_model(model: LSTMModel, test_loader: DataLoader) -> tuple[float, float]:
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for features, targets in test_loader:
            predictions.append(model(features))
            labels.append(targets)

    all_predictions = torch.cat(predictions)
    all_labels = torch.cat(labels)
    mae = torch.mean(torch.abs(all_predictions - all_labels)).item()
    rmse = torch.sqrt(torch.mean((all_predictions - all_labels) ** 2)).item()
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return mae, rmse
