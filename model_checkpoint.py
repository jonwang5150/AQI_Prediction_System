"""兩種預測模型共用的 checkpoint 載入與前處理設定驗證。"""

import pickle
from pathlib import Path

import numpy as np
import torch


def load_checkpoint_data(path: Path, *, required=()) -> dict:
    """僅載入可信任的本機模型；既有檔案含 NumPy scaler，保留相容模式。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到模型檔：{path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    fields = {"model_state_dict", "model_config", "features", "window_size",
              "scaler_mean", "scaler_scale", *required}
    if not isinstance(checkpoint, dict):
        raise ValueError("模型 checkpoint 必須為字典")
    missing = fields.difference(checkpoint)
    if missing:
        raise ValueError(f"模型 checkpoint 缺少欄位：{', '.join(sorted(missing))}")
    if not isinstance(checkpoint["model_config"], dict):
        raise ValueError("checkpoint 的 model_config 必須為字典")
    return checkpoint


def preprocessing_from_checkpoint(checkpoint):
    """統一檢查特徵順序、窗口及 scaler，回傳推論所需的四項設定。"""
    features = list(checkpoint["features"])
    window = int(checkpoint["window_size"])
    mean = np.asarray(checkpoint["scaler_mean"], dtype=float)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=float)
    if window <= 0 or not features or len(set(features)) != len(features):
        raise ValueError("checkpoint 的窗口或特徵設定無效")
    if (mean.shape != (len(features),) or scale.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or (scale <= 0).any()):
        raise ValueError("checkpoint 的 scaler 維度或數值無效")
    return features, window, mean, scale


def restore_model(model_type, checkpoint):
    """以保存的模型設定建立模型並切換為推論模式。"""
    model = model_type(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
