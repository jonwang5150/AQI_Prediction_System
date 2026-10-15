"""Load prediction data for visualization."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow direct execution from an IDE or from the visualization directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sqlalchemy import and_, or_, func, select
from datetime import timedelta

from config import settings
from db_models import AirQualityHourly, AirQualityPrediction, AirQualityPrediction8h
from update_db import database_connection


def load_prediction_data(station: str, source: str = "8h", horizon: int = 1,
                         include_future: bool = False) -> pd.DataFrame:
    """每個目標時間只取一筆；多個模型版本以最新寫入者為準。"""
    if source not in {"8h", "single"} or horizon not in range(1, 9):
        raise ValueError("source 必須為 8h 或 single，horizon 必須介於 1～8")
    if source == "8h":
        prediction = AirQualityPrediction8h
        latest = select(func.max(AirQualityHourly.observed_at)).where(
            AirQualityHourly.station == station, AirQualityHourly.aqi.is_not(None),
        ).scalar_subquery()
        selection = prediction.horizon_hours == horizon
        if include_future:
            selection = or_(
                and_(prediction.target_at <= latest, prediction.horizon_hours == horizon),
                *[and_(prediction.horizon_hours == h,
                       prediction.target_at == latest + timedelta(hours=h)) for h in range(1, 9)],
            )
        ranked = select(
            prediction.station,
            prediction.target_at,
            prediction.predicted_aqi,
            prediction.model_version,
            prediction.horizon_hours,
            func.row_number().over(
                partition_by=prediction.target_at,
                order_by=[prediction.created_at.desc(), prediction.model_version.asc()],
            ).label("rank"),
        ).where(
            prediction.station == station,
            selection,
        ).subquery()
        statement = (
            select(
                ranked.c.target_at.label("日期"),
                ranked.c.predicted_aqi.label("預測AQI"),
                AirQualityHourly.aqi.label("實際AQI"),
                ranked.c.model_version.label("模型版本"),
                ranked.c.horizon_hours.label("提前小時"),
                (ranked.c.target_at > latest).label("未來預測"),
            )
            .outerjoin(AirQualityHourly, (
                (AirQualityHourly.station == ranked.c.station)
                & (AirQualityHourly.observed_at == ranked.c.target_at)
            ))
            .where(ranked.c.rank == 1)
            .order_by(ranked.c.target_at)
        )
    else:
        statement = (
            select(
                AirQualityPrediction.predicted_at.label("日期"),
                AirQualityPrediction.predicted_aqi.label("預測AQI"),
                func.coalesce(AirQualityHourly.aqi, AirQualityPrediction.actual_aqi).label("實際AQI"),
            )
            .outerjoin(AirQualityHourly, (
                (AirQualityHourly.station == AirQualityPrediction.station)
                & (AirQualityHourly.observed_at == AirQualityPrediction.predicted_at)
            ))
            .where(AirQualityPrediction.station == station)
            .order_by(AirQualityPrediction.predicted_at)
        )
    with database_connection(settings) as connection:
        return pd.read_sql(statement, connection)


