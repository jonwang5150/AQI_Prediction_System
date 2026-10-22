#!/usr/bin/env python3
"""驗證 AQI 觀測與預測資料並寫入 PostgreSQL（Python 3.10+）。

安裝：python -m pip install "SQLAlchemy>=2.0,<2.1" "psycopg[binary]>=3,<4"
設定：修改 config.py 的資料庫連線網址與每批寫入筆數。
本模組由 crawl_to_db.py 與 predict_1h.py 呼叫，不讀寫中間檔案。

資料庫須先建立；程式會建立觀測與預測資料表。
日期保留來源資料的台灣當地時間，不進行時區轉換。
相同測站與日期會更新為最後讀到的資料（包含 NULL）。
每次寫入呼叫會使用單一交易；任一筆失敗會全部回復。
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Iterable
from contextlib import contextmanager


COLUMNS = ["測站", "日期", "AQI", "空氣品質指標", "O3", "PM2.5", "PM10", "CO", "SO2", "NO2"]
MISSING = {"", "-", "--", "NA", "N/A", "NULL", "NAN"}

MODEL_COLUMNS = ["station", "observed_at", "aqi", "air_quality_indicator", "o3", "pm25", "pm10", "co", "so2", "no2"]
PREDICTION_COLUMNS = ["測站", "日期", "預測AQI", "實際AQI"]
PREDICTION_MODEL_COLUMNS = ["station", "predicted_at", "predicted_aqi", "actual_aqi"]


def database_url_from_settings(settings):
    """使用 config.py 的連線網址，統一採用 psycopg 3 驅動。"""
    from sqlalchemy.engine import make_url
    from sqlalchemy.exc import ArgumentError

    try:
        url = make_url(settings.database_url)
        if url.drivername not in {
            "postgres", "postgresql", "postgresql+psycopg", "postgresql+psycopg2",
        }:
            raise ValueError
        if not url.host or not url.username or not url.database:
            raise ValueError
        return url.set(drivername="postgresql+psycopg")
    except (ArgumentError, TypeError, ValueError) as exc:
        raise ValueError(
            "config.py 的 database_url 必須是包含主機、帳號與資料庫名稱的 PostgreSQL 網址。"
        ) from exc


def create_database_engine(settings=None):
    """所有讀寫共用相同的 driver、連線逾時與 SQL 參數隱藏設定。"""
    from sqlalchemy import create_engine
    if settings is None:
        from config import settings
    return create_engine(database_url_from_settings(settings),
                         connect_args={"connect_timeout": 10}, hide_parameters=True)


@contextmanager
def database_connection(settings=None):
    """短期查詢用連線；成功或失敗都關閉連線並釋放 engine。"""
    engine = create_database_engine(settings)
    try:
        with engine.connect() as connection:
            yield connection
    finally:
        engine.dispose()


# 將欄位轉為有限的十進位數值，缺值回傳 None，無效值拋出錯誤。
def parse_number(value: str, column: str) -> Decimal | None:
    if value.upper() in MISSING:
        return None
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{column} 不是有效數字：{value!r}") from exc
    if not number.is_finite():
        raise ValueError(f"{column} 不是有限數字：{value!r}")
    return number


def parse_row(raw: dict[str, str]) -> dict:
    """將爬蟲取得的一列資料驗證並轉成資料庫欄位。"""
    if set(raw) != set(COLUMNS):
        raise ValueError("資料欄位與預期不符")
    row = {key: value.strip() for key, value in raw.items()}
    if not row["測站"]:
        raise ValueError("測站不可空白")
    observed_at = datetime.strptime(row["日期"], "%Y/%m/%d %H:%M")
    indicator = row["空氣品質指標"]
    values = (
        row["測站"], observed_at, parse_number(row["AQI"], "AQI"),
        None if indicator.upper() in MISSING else indicator,
        *(parse_number(row[column], column) for column in COLUMNS[4:]),
    )
    return dict(zip(MODEL_COLUMNS, values))


def parse_prediction_row(raw: dict[str, object]) -> dict:
    """將一列預測結果驗證並轉成預測資料表欄位。"""
    if set(raw) != set(PREDICTION_COLUMNS):
        raise ValueError("預測資料欄位與預期不符")

    station = str(raw["測站"]).strip()
    if not station:
        raise ValueError("測站不可空白")

    date_value = raw["日期"]
    if isinstance(date_value, datetime):
        predicted_at = date_value.replace(tzinfo=None)
    else:
        try:
            predicted_at = datetime.fromisoformat(str(date_value).strip()).replace(tzinfo=None)
        except ValueError as exc:
            raise ValueError(f"日期格式錯誤：{date_value!r}") from exc

    predicted_value = raw["預測AQI"]
    predicted_text = "" if predicted_value is None else str(predicted_value).strip()
    predicted_aqi = parse_number(predicted_text, "預測AQI")
    if predicted_aqi is None:
        raise ValueError("預測AQI不可空白")
    actual_value = raw["實際AQI"]
    actual_text = "" if actual_value is None else str(actual_value).strip()
    actual_aqi = parse_number(actual_text, "實際AQI")
    return dict(zip(
        PREDICTION_MODEL_COLUMNS,
        (station, predicted_at, predicted_aqi, actual_aqi),
    ))


def _write_rows(
    rows: Iterable[dict],
    batch_writer,
    *,
    settings,
    batch_size: int | None,
    data_name: str,
) -> int:
    """使用單一交易批次寫入資料，回傳處理筆數。"""
    if settings is None:
        from config import settings
    if batch_size is None:
        batch_size = settings.batch_size
    if batch_size <= 0:
        raise ValueError("batch_size 必須大於 0")
    try:
        from sqlalchemy.exc import SQLAlchemyError
        from sqlalchemy.orm import Session

        from db_models import Base
        import psycopg  # noqa: F401：確認 PostgreSQL 驅動已安裝
    except ImportError as exc:
        raise RuntimeError('請先安裝套件：python -m pip install "SQLAlchemy>=2.0,<2.1" "psycopg[binary]>=3,<4"') from exc

    total = 0
    engine = None
    try:
        engine = create_database_engine(settings)
        with engine.begin() as connection:
            Base.metadata.create_all(connection)
            with Session(bind=connection) as session:
                batch = []
                for row in rows:
                    batch.append(row)
                    total += 1
                    if len(batch) >= batch_size:
                        batch_writer(session, batch)
                        batch.clear()
                if batch:
                    batch_writer(session, batch)
    except SQLAlchemyError as exc:
        raise RuntimeError(
            f"PostgreSQL {data_name}寫入未完成（{type(exc).__name__}）。"
            "請檢查連線設定、資料庫是否存在，以及建表／寫入權限和資料表結構。"
        ) from exc
    finally:
        if engine is not None:
            engine.dispose()
    return total


def update_rows(rows: Iterable[dict], *, settings=None, batch_size: int | None = None) -> int:
    """在單一交易中新增或更新已驗證的觀測資料。"""

    try:
        from db_models import upsert_batch
    except ImportError as exc:
        raise RuntimeError('請先安裝套件：python -m pip install "SQLAlchemy>=2.0,<2.1" "psycopg[binary]>=3,<4"') from exc

    return _write_rows(
        rows,
        upsert_batch,
        settings=settings,
        batch_size=batch_size,
        data_name="觀測資料",
    )


def update_prediction_rows(
    rows: Iterable[dict],
    *,
    settings=None,
    batch_size: int | None = None,
) -> int:
    """在單一交易中新增或更新已驗證的預測資料。"""

    try:
        from db_models import upsert_prediction_batch
    except ImportError as exc:
        raise RuntimeError('請先安裝套件：python -m pip install "SQLAlchemy>=2.0,<2.1" "psycopg[binary]>=3,<4"') from exc

    return _write_rows(
        rows,
        upsert_prediction_batch,
        settings=settings,
        batch_size=batch_size,
        data_name="預測資料",
    )
