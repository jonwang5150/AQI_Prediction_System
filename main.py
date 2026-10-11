#!/usr/bin/env python3
"""啟動歷史 Gradio 圖表，按「更新至最新並重新預測」才爬取及執行模型。

python main.py
python main.py --station 木柵站 --port 7860 --no-browser
"""

from __future__ import annotations

import sys
from threading import Lock

import pandas as pd
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import SQLAlchemyError

import crawl_to_db
import predict_2026
import predict_2026_8h
import show_2026_prediction
from config import settings
from db_models import AirQualityHourly
from prediction_data_process import CRAWLER_COLUMNS
from update_db import database_url_from_settings


UPDATE_START = pd.Timestamp("2026-08-01")
UPDATE_LOCK = Lock()


def load_update_context(station, query_start, cutoff):
    """讀取已有觀測及歷史視窗；新下載資料由爬蟲直接交給模型。"""
    engine = create_engine(database_url_from_settings(settings),
                           connect_args={"connect_timeout": 10}, hide_parameters=True)
    try:
        with engine.connect() as connection:
            latest = connection.scalar(select(func.max(AirQualityHourly.observed_at)).where(
                AirQualityHourly.station == station,
                AirQualityHourly.observed_at <= cutoff.to_pydatetime(),
                AirQualityHourly.aqi.is_not(None),
            ))
            if latest is None:
                raise ValueError("指定測站沒有歷史 AQI 觀測，請先匯入歷史資料")
            statement = select(*[getattr(AirQualityHourly, name) for name in CRAWLER_COLUMNS]).where(
                AirQualityHourly.station == station,
                AirQualityHourly.observed_at >= query_start.to_pydatetime(),
                AirQualityHourly.observed_at <= latest,
            ).order_by(AirQualityHourly.observed_at)
            return pd.Timestamp(latest), pd.read_sql(statement, connection)
    finally:
        engine.dispose()


def update_to_latest(station: str, progress=None, now=None) -> str:
    """接續爬取、重建 8/1 起預測；重試可補完先前只成功寫入觀測的批次。"""
    if not UPDATE_LOCK.acquire(blocking=False):
        raise ValueError("目前已有更新工作執行中，請稍候")
    try:
        def report(value, message):
            if progress is not None:
                progress(value, desc=message)

        # 資料庫及來源網站使用台灣當地無時區時間。
        now = pd.Timestamp.now(tz="Asia/Taipei") if now is None else pd.Timestamp(now)
        if now.tzinfo is not None:
            now = now.tz_convert("Asia/Taipei").tz_localize(None)
        cutoff = now.floor("h")
        report(0.02, "載入 v1 模型 checkpoint")
        single_checkpoint = predict_2026.load_checkpoint(predict_2026.PROJECT_DIR / "aqi_lstm_model.pth")
        report(0.04, "載入 v2 模型 checkpoint")
        multi_checkpoint = predict_2026_8h.load_checkpoint(predict_2026_8h.PROJECT_DIR / "aqi_lstm_8h_model.pth")
        lookback = max(single_checkpoint[2], multi_checkpoint[2] + multi_checkpoint[3] - 1)
        query_start = UPDATE_START - pd.Timedelta(hours=lookback)
        report(0.06, "讀取資料庫歷史觀測與模型輸入視窗")
        previous, context = load_update_context(station, query_start, cutoff)
        crawl_start = max(UPDATE_START, previous + pd.Timedelta(hours=1))
        downloaded = []
        if crawl_start <= cutoff:
            report(0.10, f"下載 {crawl_start} 至 {cutoff} 已發布的小時資料")
            downloaded = crawl_to_db.main([
                "--stations", station, "--start", str(crawl_start.date()),
                "--end", str(cutoff.date()), "--no-dry-run",
            ], observed_after=crawl_start - pd.Timedelta(hours=1), observed_until=cutoff,
                station_names={station.removesuffix("站"): station})

        # 同日下載可能含重疊列，以新資料優先；不再把新下載資料從 DB 讀回。
        report(0.45, "合併觀測資料並確認最新有效時間")
        observations = pd.concat([context, pd.DataFrame(downloaded)], ignore_index=True)
        observations["observed_at"] = pd.to_datetime(observations["observed_at"])
        observations = observations.loc[(observations["station"] == station)
                                         & (observations["observed_at"] <= cutoff)].copy()
        observations = observations.drop_duplicates(["station", "observed_at"], keep="last")
        valid_aqi = pd.to_numeric(observations["aqi"], errors="coerce").notna()
        latest = observations.loc[valid_aqi, "observed_at"].max()
        if pd.isna(latest) or latest < UPDATE_START - pd.Timedelta(hours=1):
            raise ValueError("尚無足夠的新觀測可接續 2026-08-01 的預測")
        observations = observations.loc[observations["observed_at"] <= latest].copy()
        report(0.50, "執行 v1 單步模型")
        if latest >= UPDATE_START:
            predict_2026.main(data=observations, argv=[
                "--station", station, "--start", str(UPDATE_START),
                "--end", str(latest + pd.Timedelta(hours=1)),
            ])
        report(0.72, "執行 v2 歷史回測與未來 8 小時預測")
        predict_2026_8h.main(data=observations, argv=[
            "--station", station, "--start", str(UPDATE_START),
            "--end", str(latest + pd.Timedelta(hours=9)), "--latest-origin", str(latest),
        ])
        report(0.95, "重新載入圖表與評估指標")
        return (f"更新完成：本次取得 {len(downloaded):,} 筆觀測。實際觀測截至 **{latest}**（台灣時間）；"
                f"8h 預測延伸至 **{latest + pd.Timedelta(hours=8)}**。"
                "網站尚未發布的小時不會當成真實觀測；無真實值的未來預測不納入評估。")
    finally:
        UPDATE_LOCK.release()


def main(argv=None) -> int:
    # 使用專案實際檔名 show_2026_prediction.py，維持原有參數及四種圖表。
    args = show_2026_prediction.build_parser().parse_args(argv)
    print(f"Python 執行環境：{sys.executable}", flush=True)
    return show_2026_prediction.main(
        argv=argv, update_callback=lambda progress: update_to_latest(args.station.strip(), progress))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, SQLAlchemyError) as exc:
        print(f"無法啟動（{type(exc).__name__}），請檢查資料庫連線及預測資料。", file=sys.stderr)
        raise SystemExit(1)
