"""一次輸出未來多小時 AQI 的 LSTM 模型。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class MultiHorizonLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 7,
        forecast_horizon: int = 8,
        hidden_size: int = 64,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, forecast_horizon),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(inputs)
        return self.fc(output[:, -1, :])


def save_checkpoint_new(
    model: MultiHorizonLSTM,
    path: str | Path,
    features: list[str],
    window_size: int,
    forecast_horizon: int,
    scaler: Any,
) -> None:
    """保存權重以及多步推論所需的前處理設定。"""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_size": len(features),
            "forecast_horizon": forecast_horizon,
            "hidden_size": 64,
            "num_layers": 1,
        },
        "features": features,
        "window_size": window_size,
        "forecast_horizon": forecast_horizon,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }
    torch.save(checkpoint, path)
    print(f"模型已儲存：{Path(path).resolve()}")
