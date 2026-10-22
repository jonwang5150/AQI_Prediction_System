#!/usr/bin/env python3
"""啟動歷史 Gradio 圖表，按「更新至最新並重新預測」才爬取及執行模型。

python main.py
python main.py --station 木柵站 --port 7860 --no-browser
"""

from __future__ import annotations

import sys
from threading import Lock

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

import crawl_to_db
import predict_1h
import predict_8h
from prediction_visualization import visualizations
from build_pgvector import build_knowledge_base
from config import settings
from db_models import AirQualityHourly, AirQualityPrediction, AirQualityPrediction8h
from prediction_data_process import read_database_observations
from update_db import database_connection


UPDATE_LOCK = Lock()


def load_update_context(station, lookback, cutoff):
    """依觀測與預測進度決定增量區間，只讀待預測資料及必要歷史視窗。"""
    with database_connection(settings) as connection:
        latest = connection.scalar(select(func.max(AirQualityHourly.observed_at)).where(
            AirQualityHourly.station == station,
            AirQualityHourly.observed_at <= cutoff.to_pydatetime(),
        ))
        if latest is None:
            raise ValueError("指定測站沒有歷史 AQI 觀測，請先匯入歷史資料")
        latest = pd.Timestamp(latest)
        single_last = connection.scalar(select(func.max(AirQualityPrediction.predicted_at)).where(
            AirQualityPrediction.station == station,
            AirQualityPrediction.predicted_at <= latest,
        ))
        complete_targets = select(AirQualityPrediction8h.target_at).where(
            AirQualityPrediction8h.station == station,
            AirQualityPrediction8h.target_at <= latest,
        ).group_by(AirQualityPrediction8h.target_at, AirQualityPrediction8h.model_version).having(
            func.count(func.distinct(AirQualityPrediction8h.horizon_hours)) == 8,
        ).subquery()
        multi_last = connection.scalar(select(func.max(complete_targets.c.target_at)))
        starts = {
            "single": pd.Timestamp(single_last if single_last is not None else latest) + pd.Timedelta(hours=1),
            "8h": pd.Timestamp(multi_last if multi_last is not None else latest) + pd.Timedelta(hours=1),
        }
        query_start = min(starts.values()) - pd.Timedelta(hours=lookback)
        context = read_database_observations(
            connection, station, query_start, latest, model_columns=False)
        return latest, context, starts


def update_to_latest(station: str, progress=None, now=None) -> str:
    """由最後觀測的下一小時爬取；各模型只補算尚未完成的目標時段。"""
    if not UPDATE_LOCK.acquire(blocking=False):
        raise ValueError("目前已有更新工作執行中，請稍候")
    try:
        # 將更新進度傳給介面；未提供回呼時略過通知。
        def report(value, message):
            if progress is not None:
                progress(value, desc=message)

        # 資料庫及來源網站使用台灣當地無時區時間。
        now = pd.Timestamp.now(tz="Asia/Taipei") if now is None else pd.Timestamp(now)
        if now.tzinfo is not None:
            now = now.tz_convert("Asia/Taipei").tz_localize(None)
        cutoff = now.floor("h")
        report(0.02, "載入 v1 模型 checkpoint")
        single_checkpoint = predict_1h.load_checkpoint(predict_1h.PROJECT_DIR / "aqi_lstm_model.pth")
        report(0.04, "載入 v2 模型 checkpoint")
        multi_checkpoint = predict_8h.load_checkpoint(predict_8h.PROJECT_DIR / "aqi_lstm_8h_model.pth")
        lookback = max(single_checkpoint[2], multi_checkpoint[2] + multi_checkpoint[3] - 1)
        report(0.06, "讀取資料庫歷史觀測與模型輸入視窗")
        previous, context, starts = load_update_context(station, lookback, cutoff)
        crawl_start = previous + pd.Timedelta(hours=1)
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
        if pd.isna(latest):
            raise ValueError("模型輸入視窗內沒有有效 AQI 觀測")
        if all(start > latest for start in starts.values()):
            return (f"目前沒有新的有效 AQI 觀測需要預測。資料表最後時間：**{previous}**；"
                    f"有效 AQI 截至 **{latest}**（台灣時間）。本次取得 {len(downloaded):,} 筆觀測，保留既有預測。")
        observations = observations.loc[observations["observed_at"] <= latest].copy()
        report(0.50, "執行 v1 單步模型")
        if latest >= starts["single"]:
            predict_1h.main(data=observations, argv=[
                "--station", station, "--start", str(starts["single"]),
                "--end", str(latest + pd.Timedelta(hours=1)),
            ])
        report(0.72, "執行 v2 歷史回測與未來 8 小時預測")
        if latest >= starts["8h"]:
            predict_8h.main(data=observations, argv=[
                "--station", station, "--start", str(starts["8h"]),
                "--end", str(latest + pd.Timedelta(hours=9)), "--latest-origin", str(latest),
            ])
        report(0.90, "更新向量知識庫")
        knowledge_document_count = build_knowledge_base()
        report(0.98, "重新載入圖表與評估指標")
        return (f"更新完成：本次取得 {len(downloaded):,} 筆觀測。實際觀測截至 **{latest}**（台灣時間）；"
                f"8h 預測延伸至 **{latest + pd.Timedelta(hours=8)}**。"
                f"向量知識庫已更新 {knowledge_document_count:,} 份文件。"
                "網站尚未發布的小時不會當成真實觀測；無真實值的未來預測不納入評估。")
    finally:
        UPDATE_LOCK.release()


# 解析啟動參數並開啟儀表板，接上資料更新與預測功能。
def main(argv=None) -> int:
    # 使用 prediction_visualization.visualizations，維持原有參數及四種圖表。
    args = visualizations.build_parser().parse_args(argv)
    print(f"Python 執行環境：{sys.executable}", flush=True)
    return visualizations.launch_dashboard(
        args, update_callback=lambda progress: update_to_latest(args.station.strip(), progress))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, SQLAlchemyError) as exc:
        print(f"無法啟動（{type(exc).__name__}），請檢查資料庫連線及預測資料。", file=sys.stderr)
        raise SystemExit(1)
