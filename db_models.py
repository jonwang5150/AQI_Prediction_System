"""AQI 資料的 SQLAlchemy 2.x ORM 模型與 PostgreSQL 批次寫入。"""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import CheckConstraint, DateTime, Integer, Numeric, Text, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


# 提供所有 SQLAlchemy 資料表模型共用的宣告基底。
class Base(DeclarativeBase):
    pass


# 對應逐時空氣品質觀測表，以測站與觀測時間識別資料。
class AirQualityHourly(Base):
    __tablename__ = "air_quality_hourly"
    __table_args__ = {"schema": "public"}

    station: Mapped[str] = mapped_column(Text, primary_key=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), primary_key=True)
    aqi: Mapped[Decimal | None] = mapped_column(Numeric)
    air_quality_indicator: Mapped[str | None] = mapped_column(Text)
    o3: Mapped[Decimal | None] = mapped_column(Numeric)
    pm25: Mapped[Decimal | None] = mapped_column(Numeric)
    pm10: Mapped[Decimal | None] = mapped_column(Numeric)
    co: Mapped[Decimal | None] = mapped_column(Numeric)
    so2: Mapped[Decimal | None] = mapped_column(Numeric)
    no2: Mapped[Decimal | None] = mapped_column(Numeric)


class AirQualityPrediction(Base):
    """逐時 AQI 預測；同一測站、同一時間只保留最新結果。"""

    __tablename__ = "air_quality_predictions"
    __table_args__ = {"schema": "public"}

    station: Mapped[str] = mapped_column(Text, primary_key=True)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), primary_key=True)
    predicted_aqi: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    actual_aqi: Mapped[Decimal | None] = mapped_column(Numeric)


class AirQualityPrediction8h(Base):
    """同一目標時間保留各提前小時與模型版本，不重複儲存真實 AQI。"""

    __tablename__ = "air_quality_predictions_8h"
    __table_args__ = (
        CheckConstraint("horizon_hours BETWEEN 1 AND 8", name="ck_prediction_8h_horizon"),
        {"schema": "public"},
    )

    station: Mapped[str] = mapped_column(Text, primary_key=True)
    target_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), primary_key=True)
    horizon_hours: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version: Mapped[str] = mapped_column(Text, primary_key=True)
    predicted_aqi: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


def _upsert_model_batch(
    session: Session, rows: list[dict], model: type[Base]
) -> None:
    """依模型主鍵去重並批次 upsert；衝突時更新所有非主鍵欄位，不提交交易。"""
    if not rows:
        return
    table = model.__table__
    keys = [column.name for column in table.primary_key.columns]
    # PostgreSQL 單一 INSERT 不可更新同一鍵兩次；保留每個鍵最後讀到的資料。
    unique = {tuple(row[key] for key in keys): row for row in rows}
    values = list(unique.values())
    # 控制單次 SQL 的參數數量，即使使用者指定較大的 batch-size 也可執行。
    for offset in range(0, len(values), 1000):
        statement = insert(model).values(values[offset:offset + 1000])
        statement = statement.on_conflict_do_update(
            index_elements=keys,
            set_={
                column.name: statement.excluded[column.name]
                for column in table.columns
                if not column.primary_key
            },
        )
        session.execute(statement)


# 批次新增觀測資料，主鍵衝突時更新既有紀錄。
def upsert_batch(session: Session, rows: list[dict]) -> None:
    _upsert_model_batch(session, rows, AirQualityHourly)


# 批次新增或更新單步 AQI 預測紀錄。
def upsert_prediction_batch(session: Session, rows: list[dict]) -> None:
    _upsert_model_batch(session, rows, AirQualityPrediction)


# 批次新增或更新多步預測，保留各提前小時及模型版本。
def upsert_prediction_8h_batch(session: Session, rows: list[dict]) -> None:
    _upsert_model_batch(session, rows, AirQualityPrediction8h)
