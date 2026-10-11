"""AQI 未來預測結果的互動式視覺化。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import and_, or_, create_engine, func, select
from datetime import timedelta

from config import settings
from db_models import AirQualityHourly, AirQualityPrediction, AirQualityPrediction8h
from update_db import database_url_from_settings


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
    engine = create_engine(
        database_url_from_settings(settings),
        connect_args={"connect_timeout": 10},
        hide_parameters=True,
    )
    try:
        with engine.connect() as connection:
            return pd.read_sql(statement, connection)
    finally:
        engine.dispose()


def evaluate_prediction_data(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """以所有模型共同具有有限預測與真實值的時間點評估，不補值。"""
    aligned = {}
    common = None
    for source, data in datasets.items():
        frame = data.set_index("日期")[["預測AQI", "實際AQI"]].apply(
            pd.to_numeric, errors="coerce")
        if frame.index.has_duplicates:
            raise ValueError("評估資料的目標時間不可重複")
        frame = frame.loc[np.isfinite(frame.to_numpy(dtype=float)).all(axis=1)]
        aligned[source] = frame
        common = frame.index if common is None else common.intersection(frame.index)
    rows = []
    for source, frame in aligned.items():
        paired = frame.loc[common]
        errors = (paired["預測AQI"] - paired["實際AQI"]).to_numpy(dtype=float)
        count = len(errors)
        rows.append({
            "來源": source, "有效筆數": count,
            "未納入筆數": len(datasets[source]) - count,
            "MAE": float(np.abs(errors).mean()) if count else np.nan,
            "RMSE": float(np.sqrt(np.mean(errors ** 2))) if count else np.nan,
            "Bias": float(errors.mean()) if count else np.nan,
            "評估開始": common.min() if count else pd.NaT,
            "評估結束": common.max() if count else pd.NaT,
        })
    return pd.DataFrame(rows)


VIEW_LABELS = {
    "single": "v1｜單步模型 vs 真實數據",
    "8h": "v2｜8h 模型 vs 真實數據",
    "both": "三線比較｜v1、v2、真實數據",
    "metrics": "評估比較｜MAE / RMSE / Bias",
}
MODEL_COLORS = {"single": "#2563eb", "8h": "#16a34a"}

DASHBOARD_CSS = """
#aqi-refresh {width: 144px !important; min-width: 144px !important;
              max-width: 144px !important; flex: 0 0 144px !important;
              height: 36px !important; min-height: 36px !important; padding: 6px 12px !important;}
#aqi-busy {padding: 24px !important; border: 1px solid #fdba74;
           border-radius: 12px; background: var(--block-background-fill);}
#aqi-progress {min-height: 140px !important;}
"""


def build_prediction_figures(
    datasets: dict[str, pd.DataFrame],
    station: str,
    initial_window_days: int = 7,
    horizon: int = 1,
) -> tuple[dict, pd.DataFrame, str]:
    """建立獨立圖表；純資料運算，無資料庫或瀏覽器副作用。"""
    import plotly.graph_objects as go

    if initial_window_days <= 0:
        raise ValueError("initial_window_days 必須大於 0")
    if horizon not in range(1, 9):
        raise ValueError("horizon 必須介於 1～8")
    if not datasets or set(datasets) - {"single", "8h"}:
        raise ValueError("資料來源必須為 single 或 8h")
    cleaned = {}
    for source, data in datasets.items():
        data = data.copy()
        data["日期"] = pd.to_datetime(data["日期"], errors="coerce")
        for column in ["預測AQI", "實際AQI"]:
            data[column] = pd.to_numeric(data[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan)
        data = data.dropna(subset=["日期"]).sort_values("日期")
        if data.empty or not data["預測AQI"].notna().any():
            raise ValueError(f"測站「{station}」的 {source} 來源沒有可繪製的預測資料")
        cleaned[source] = data

    labels = {"single": "v1 單步模型", "8h": f"v2 8h 模型（h{horizon}）"}
    metrics = evaluate_prediction_data(cleaned)
    summary = metrics.copy()
    summary["來源"] = summary["來源"].map(labels)
    first = metrics.iloc[0]
    count = int(first["有效筆數"])
    period = (f"{first['評估開始']} ～ {first['評估結束']}"
              if count else "無共同有效資料，指標為 N/A")
    note = (
        f"**評估期間：** {period}；每個模型 **{count} 筆**。\n\n"
        "使用所有已載入模型共同具有有效預測與真實值的時間點，不補值。"
        "指標以全期間計算，不隨趨勢圖縮放更新。\n\n"
        "MAE、RMSE 越小越好；Bias = 預測 − 實際，正值高估、負值低估。"
    )
    figures = {}
    views = list(cleaned)
    if len(cleaned) == 2:
        views.append("both")
    for view in views:
        selected = list(cleaned) if view == "both" else [view]
        figure = go.Figure()
        actual = pd.concat([cleaned[key][["日期", "實際AQI"]] for key in selected])
        actual = actual.groupby("日期", as_index=False)["實際AQI"].first().sort_values("日期")
        figure.add_trace(go.Scatter(
            x=actual["日期"], y=actual["實際AQI"], mode="lines",
            name="真實 AQI", line={"color": "#f97316", "width": 2},
            connectgaps=False,
        ))
        for key in selected:
            data = cleaned[key]
            future = data.loc[data.get("未來預測", pd.Series(False, index=data.index)).fillna(False).astype(bool)]
            if not future.empty:
                data = data.loc[~data.index.isin(future.index)]
            figure.add_trace(go.Scatter(
                x=data["日期"], y=data["預測AQI"], mode="lines",
                name=labels[key], line={"color": MODEL_COLORS[key], "width": 1.8},
                connectgaps=False,
                customdata=data[["模型版本"]].to_numpy() if "模型版本" in data else None,
                hovertemplate=("%{x}<br>預測 AQI：%{y:.2f}<br>模型：%{customdata[0]}<extra>%{fullData.name}</extra>"
                               if "模型版本" in data else None),
            ))
            if not future.empty:
                figure.add_trace(go.Scatter(
                    x=future["日期"], y=future["預測AQI"], mode="lines+markers",
                    name="v2 最新一輪未來 8 小時", line={"color": MODEL_COLORS[key], "dash": "dash", "width": 2},
                    customdata=future[["提前小時"]].to_numpy(),
                    hovertemplate="%{x}<br>預測 AQI：%{y:.2f}<br>提前 %{customdata[0]} 小時<extra>尚無真實值</extra>",
                ))
                origin = future["日期"].min() - pd.Timedelta(hours=1)
                figure.add_shape(type="rect", x0=origin, x1=future["日期"].max(),
                                 y0=0, y1=1, yref="paper", line_width=0,
                                 fillcolor="rgba(22,163,74,0.07)", layer="below")
        last_time = actual["日期"].max()
        first_visible = max(actual["日期"].min(), last_time - pd.Timedelta(days=initial_window_days))
        figure.update_layout(
            title=f"{station}｜{VIEW_LABELS[view]}", template="plotly_white",
            height=620, hovermode="x unified", xaxis_title="目標時間", yaxis_title="AQI",
            legend={"orientation": "h", "y": 1.12, "x": 0},
            margin={"l": 60, "r": 25, "t": 130, "b": 50},
        )
        figure.update_xaxes(
            range=[first_visible, last_time],
            rangeslider={"visible": True, "thickness": 0.08},
            rangeselector={"buttons": [
                {"count": 1, "label": "1 天", "step": "day", "stepmode": "backward"},
                {"count": 7, "label": "7 天", "step": "day", "stepmode": "backward"},
                {"count": 1, "label": "1 個月", "step": "month", "stepmode": "backward"},
                {"step": "all", "label": "全部"},
            ]},
        )
        figures[view] = figure

    figure = go.Figure()
    for row in metrics.to_dict(orient="records"):
        key = row["來源"]
        figure.add_trace(go.Bar(
            x=["MAE", "RMSE", "Bias"],
            y=[row[metric] if pd.notna(row[metric]) else None for metric in ["MAE", "RMSE", "Bias"]],
            name=labels[key], marker_color=MODEL_COLORS[key],
            texttemplate="%{y:.3f}", textposition="auto",
            hovertemplate="%{x}：%{y:.4f}<extra>%{fullData.name}</extra>",
        ))
    figure.update_layout(
        title=f"{station}｜預測誤差比較", template="plotly_white", height=560,
        barmode="group", yaxis_title="誤差（AQI）",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"l": 60, "r": 25, "t": 110, "b": 50},
    )
    figure.update_yaxes(zeroline=True, zerolinecolor="#64748b", zerolinewidth=1)
    if not count:
        figure.add_annotation(text="無共同有效資料，無法計算評估指標（N/A）",
                              x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    figures["metrics"] = figure
    return figures, summary, note


def build_prediction_app(
    datasets: dict[str, pd.DataFrame],
    station: str = "木柵站",
    initial_window_days: int = 7,
    horizon: int = 1,
    update_callback=None,
    reload_callback=None,
):
    """以四個按鈕更新同一個 Plot，評估明細僅在評估畫面顯示。"""
    from functools import partial
    import inspect
    import gradio as gr

    figures, summary, note = build_prediction_figures(
        datasets, station, initial_window_days, horizon)
    initial_view = "both" if "both" in figures else next(iter(datasets))
    # Gradio 6 的 CSS 移至 launch；Gradio 5 則在 Blocks 設定。
    style_options = {} if "css" in inspect.signature(gr.Blocks.launch).parameters else {"css": DASHBOARD_CSS}
    with gr.Blocks(title=f"{station} AQI 預測比較", analytics_enabled=False, **style_options) as app:
        gr.Markdown(f"# {station} AQI 預測比較\n按下按鈕切換圖表，每次呈現一張。")
        gr.Markdown(f"8h 模型採用 **h{horizon}（提前 {horizon} 小時）** 的預測；趨勢圖可縮放與拖曳時間軸。")
        if update_callback is not None:
            gr.Markdown("歷史曲線使用指定 h；綠色虛線為最新觀測基準的未來 h1～h8，不納入誤差評估。")
            with gr.Row():
                refresh = gr.Button("更新至最新", variant="primary", size="sm",
                                    scale=0, min_width=144, elem_id="aqi-refresh")
            status = gr.Markdown("目前顯示資料庫中的預測。按下更新後，將爬取已發布觀測並重新預測。")
            with gr.Column(visible=False, elem_id="aqi-busy") as busy:
                gr.Markdown("### 正在更新資料與預測\n請稍候，完成後會自動顯示圖表。")
                busy_progress = gr.Markdown("正在準備更新…", elem_id="aqi-progress")
        with gr.Column() as chart_area:
            with gr.Row():
                buttons = {
                    key: gr.Button(label, interactive=key in figures,
                                   variant="primary" if key == initial_view else "secondary")
                    for key, label in VIEW_LABELS.items()
                }
            if len(datasets) < 2:
                gr.Markdown("目前只載入一個模型；以 `--source both` 啟動可比較兩個模型。")
            plot = gr.Plot(value=figures[initial_view], show_label=False)
            with gr.Column(visible=False) as details:
                note_component = gr.Markdown(note)
                table = gr.Dataframe(value=summary, interactive=False, label="評估明細")

        def select_view(view):
            return (
                figures[view], gr.update(visible=view == "metrics"),
                *[gr.update(variant="primary" if key == view else "secondary") for key in buttons],
            )

        for key, button in buttons.items():
            if key in figures:
                button.click(partial(select_view, key), inputs=None,
                             outputs=[plot, details, *buttons.values()],
                             queue=False, api_name=f"view_{key}")
        if update_callback is not None:
            def refresh_data(progress=gr.Progress()):
                nonlocal figures, summary, note
                stage = "開始更新"

                def report_progress(value, desc=""):
                    nonlocal stage
                    if desc:
                        stage = desc
                    progress(value, desc=desc)

                try:
                    message = update_callback(report_progress)
                    stage = "讀取更新後的預測資料"
                    new_datasets = reload_callback()
                    stage = "建立更新後的圖表與評估指標"
                    new_figures, new_summary, new_note = build_prediction_figures(
                        new_datasets, station, initial_window_days, horizon)
                    figures, summary, note = new_figures, new_summary, new_note
                    return (*select_view(initial_view), note, summary, message)
                except Exception as exc:
                    # 不把可能含有資料庫連線資訊的原始錯誤送到瀏覽器。
                    import logging
                    import traceback
                    from crawl_to_db import CrawlBatchError
                    reason = str(exc).strip() if isinstance(exc, (ValueError, CrawlBatchError)) else "請查看終端機中的錯誤位置"
                    reason = reason or "例外未提供詳細訊息，請查看終端機中的錯誤位置"
                    # 保留每一層的錯誤位置；資料庫原始 SQL／參數不寫進紀錄。
                    chain = []
                    current = exc
                    seen = set()
                    while current is not None and id(current) not in seen:
                        seen.add(id(current))
                        detail = (str(current) if isinstance(current, (ValueError, CrawlBatchError, ImportError))
                                  else "")
                        chain.append(f"{type(current).__name__}: {detail}\n"
                                     + "".join(traceback.format_tb(current.__traceback__)))
                        current = current.__cause__
                    logging.getLogger(__name__).error(
                        "更新流程失敗 [%s] %s: %s\n%s", stage, type(exc).__name__, reason,
                        "\n原因鏈：\n".join(chain),
                    )
                    message = (f"更新未完成（{type(exc).__name__}）。\n\n"
                               f"**失敗步驟：{stage}**\n\n**原因：** {reason}\n\n保留目前圖表；"
                               "已成功寫入的批次會保留，可再次按更新重試。")
                    return (*[gr.skip() for _ in range(2 + len(buttons) + 2)], message)

            def begin_update():
                return (gr.update(value="更新中…", interactive=False),
                        gr.update(visible=False), gr.update(visible=True), gr.update(visible=False))

            def finish_update():
                return (gr.update(value="更新至最新", interactive=True),
                        gr.update(visible=True), gr.update(visible=False), gr.update(visible=True))

            lifecycle_outputs = [refresh, chart_area, busy, status]
            refresh.click(begin_update, inputs=None, outputs=lifecycle_outputs,
                          queue=False, show_progress="hidden", api_name=False).then(
                          refresh_data, inputs=None,
                          outputs=[plot, details, *buttons.values(), note_component, table, status],
                          concurrency_limit=1, concurrency_id="aqi-update", trigger_mode="once",
                          show_progress="full", show_progress_on=[busy_progress],
                          api_name=False).then(finish_update, inputs=None, outputs=lifecycle_outputs,
                                               queue=False, show_progress="hidden", api_name=False)
    return app


def show_prediction_visualization(
    station: str = "木柵站",
    initial_window_days: int = 7,
    source: str = "both",
    horizon: int = 1,
    server_port: int | None = None,
    inbrowser: bool = True,
    update_callback=None,
) -> None:
    """從資料庫載入預測，啟動本機 Gradio 網頁，直到 Ctrl+C 結束。"""
    if source not in {"both", "8h", "single"}:
        raise ValueError("source 必須為 both、8h 或 single")
    if initial_window_days <= 0 or horizon not in range(1, 9):
        raise ValueError("window-days 必須大於 0，horizon 必須介於 1～8")
    sources = ["single", "8h"] if source == "both" else [source]
    def reload_data():
        return {key: load_prediction_data(station, source=key, horizon=horizon,
                                         include_future=update_callback is not None) for key in sources}
    datasets = reload_data()
    app = build_prediction_app(datasets, station, initial_window_days, horizon,
                               update_callback=update_callback, reload_callback=reload_data)
    app.queue()
    import inspect
    style_options = {"css": DASHBOARD_CSS} if "css" in inspect.signature(app.launch).parameters else {}
    app.launch(server_name="127.0.0.1", server_port=server_port,
               inbrowser=inbrowser, share=False, **style_options)
