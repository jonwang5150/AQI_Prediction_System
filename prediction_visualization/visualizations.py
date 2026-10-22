"""AQI 未來預測結果的互動式視覺化。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from .evaluate_prediction_data import evaluate_prediction_data
from .load_prediction_data import load_prediction_data


VIEW_LABELS = {
    "single": "v1｜單步模型 vs 真實數據",
    "8h": "v2｜8h 模型 vs 真實數據",
    "both": "三線比較｜v1、v2、真實數據",
    "metrics": "評估比較｜MAE / RMSE / Bias",
}
MODEL_COLORS = {"single": "#2563eb", "8h": "#16a34a"}

DASHBOARD_CSS = Path(__file__).with_name("dashboard.css").read_text(encoding="utf-8")


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


def build_agent_panel():
    """以 session State 保存 agent 訊息，顯示最終回答與已產生的圖表。"""
    import inspect
    import gradio as gr

    # 處理聊天送出事件，更新對話、參考來源、圖表與按鈕狀態。
    def respond(question, display_history, agent_history):
        from agent.agent import chat_with_agent
        display_history = list(display_history or [])
        if not question or not question.strip():
            yield (gr.skip(), display_history, agent_history, gr.skip(), gr.skip(),
                   gr.update(interactive=True), gr.update(interactive=True), gr.skip(), gr.skip(), gr.skip())
            return
        display_history.append({'role': 'user', 'content': question.strip()})
        display_history.append({'role': 'assistant', 'content': '正在處理你的需求，請稍候…'})
        yield (gr.update(value='', interactive=False), display_history, agent_history, gr.skip(), '',
               gr.update(interactive=False), gr.update(interactive=False), gr.skip(), gr.skip(), gr.skip())
        result = chat_with_agent(question, agent_history)
        display_history[-1] = {'role': 'assistant', 'content': result['answer']}
        has_charts = bool(result['charts'])
        yield (gr.update(value='', interactive=True), display_history, result['history'],
               result['charts'] if has_charts else gr.skip(), result['references'],
               gr.update(interactive=True), gr.update(interactive=True),
               gr.update(visible=True) if has_charts else gr.skip(),
               gr.update(elem_classes=['aqi-agent-wide']) if has_charts else gr.skip(),
               gr.update(visible=True) if has_charts else gr.skip())

    with gr.Column(visible=False, elem_id="aqi-rag-panel") as panel:
        gr.Markdown("### 木柵 AQI 智能助理")
        gr.Markdown(
            "查詢木柵觀測站的 AQI、歷史預測表現與木柵天氣，也能把資料畫成圖。"
            "可接續追問，助理會沿用本次對話中已確認的條件。",
            elem_id="aqi-rag-description")
        with gr.Accordion("可以問什麼？", open=False):
            gr.Markdown(
                "- **日期／月份摘要**：查詢 2026 年每日或每月的平均 AQI、MAE、RMSE、Bias。\n"
                "- **原始觀測與預測**：查詢資料庫已有的木柵逐時資料；8h 預測需指定提前小時與模型版本。\n"
                "- **資料繪圖**：將某月每日平均 AQI 畫成趨勢圖，或依你提供的數據製圖。\n"
                "- **木柵天氣**：查目前天氣及今天起最多 7 天的預報。\n\n"
                "服務僅限木柵；評估摘要的平均採有效預測配對樣本，可能與全部觀測平均不同。"
                "助理不會自動套用上方圖表的時間範圍、提前小時或模型版本，請在問題中說明。")
        chatbot_options = {'type': 'messages'} if 'type' in inspect.signature(gr.Chatbot).parameters else {}
        with gr.Row(elem_id="aqi-agent-layout"):
            with gr.Column(scale=5, min_width=320):
                chatbot = gr.Chatbot(label="對話", height=400, elem_id="aqi-agent-chat", **chatbot_options)
                agent_state = gr.State([])
                question = gr.Textbox(label="你的問題", placeholder="輸入日期、月份或繪圖需求；可接續上一則回答追問。",
                                      lines=2, max_lines=4)
                with gr.Row(elem_id="aqi-rag-actions"):
                    submit = gr.Button("送出訊息", variant="primary", size="sm", scale=0,
                                       min_width=132, elem_classes=["aqi-rag-button"])
                    clear = gr.Button("清除對話", size="sm", scale=0, elem_classes=["aqi-rag-button"])
                    show_chart = gr.Button("查看圖表", size="sm", scale=0,
                                           elem_classes=["aqi-rag-button"])
                with gr.Accordion("提問範例", open=False):
                    gr.Examples(examples=[
                        ["2026/4/9 的平均 AQI 是多少？"],
                        ["比較 2026 年 5 月與 6 月的預測 MAE。"],
                        ["將 2026 年 6 月每日平均 AQI 畫成趨勢圖。"],
                        ["木柵明天會下雨嗎？"],
                    ], inputs=question, label="點選範例後即可送出")
                with gr.Accordion("查看參考摘要", open=False):
                    sources = gr.Textbox(label="本次檢索依據", lines=8, max_lines=12,
                                         interactive=False, elem_id="aqi-rag-sources")
            with gr.Column(scale=7, min_width=360, visible=False, elem_id="aqi-agent-chart-column") as chart_column:
                gallery = gr.Gallery(label="最近產生的圖表", columns=1, type="filepath", interactive=False,
                                     object_fit="contain", preview=True, height=580, elem_id="aqi-agent-gallery")
                hide_chart = gr.Button("收起圖表", size="sm", scale=0)
    gr.on(
        triggers=[submit.click, question.submit], fn=respond,
        inputs=[question, chatbot, agent_state],
        outputs=[question, chatbot, agent_state, gallery, sources, submit, clear, chart_column, panel, show_chart],
        concurrency_limit=1, concurrency_id="aqi-agent", trigger_mode="once",
        show_progress="full", show_progress_on=[chatbot], api_name=False,
    )
    clear.click(lambda: ('', [], [], [], '', gr.update(visible=False), gr.update(elem_classes=[]), gr.update(visible=True)), inputs=None,
                outputs=[question, chatbot, agent_state, gallery, sources, chart_column, panel, show_chart],
                concurrency_limit=1, concurrency_id="aqi-agent", api_name=False)
    hide_chart.click(lambda: (gr.update(visible=False), gr.update(elem_classes=[])), inputs=None,
                     outputs=[chart_column, panel], queue=False, api_name=False)
    show_chart.click(lambda: (gr.update(visible=True), gr.update(elem_classes=['aqi-agent-wide'])), inputs=None,
                     outputs=[chart_column, panel], queue=False, api_name=False)
    return panel


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
        gr.Markdown(f"# {station} AQI 預測比較\n按下功能按鈕切換圖表、評估或智能助理。")
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
                buttons["rag"] = gr.Button("智能助理", variant="secondary")
            with gr.Column() as chart_content:
                if len(datasets) < 2:
                    gr.Markdown("目前只載入一個模型；以 `--source both` 啟動可比較兩個模型。")
                plot = gr.Plot(value=figures[initial_view], show_label=False)
                with gr.Column(visible=False) as details:
                    note_component = gr.Markdown(note)
                    table = gr.Dataframe(value=summary, interactive=False, label="評估明細")
            rag_panel = build_agent_panel()

        view_outputs = [plot, details, *buttons.values(), chart_content, rag_panel]

        # 切換圖表、評估表或助理面板，並同步導覽按鈕樣式。
        def select_view(view):
            return (
                gr.skip() if view == "rag" else figures[view], gr.update(visible=view == "metrics"),
                *[gr.update(variant="primary" if key == view else "secondary") for key in buttons],
                gr.update(visible=view != "rag"), gr.update(visible=view == "rag"),
            )

        for key, button in buttons.items():
            if key in figures or key == "rag":
                button.click(partial(select_view, key), inputs=None,
                             outputs=view_outputs,
                             queue=False, api_name=f"view_{key}")
        if update_callback is not None:
            # 執行資料更新後重新載入圖表，失敗時顯示所處階段。
            def refresh_data(progress=gr.Progress()):
                nonlocal figures, summary, note
                stage = "開始更新"

                # 記錄目前更新階段，並將進度傳給 Gradio 介面。
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
                    return (*[gr.skip() for _ in range(len(view_outputs) + 2)], message)

            # 更新開始時停用按鈕並切換為進度顯示狀態。
            def begin_update():
                return (gr.update(value="更新中…", interactive=False),
                        gr.update(visible=False), gr.update(visible=True), gr.update(visible=False))

            # 更新結束後恢復按鈕與一般操作介面。
            def finish_update():
                return (gr.update(value="更新至最新", interactive=True),
                        gr.update(visible=True), gr.update(visible=False), gr.update(visible=True))

            lifecycle_outputs = [refresh, chart_area, busy, status]
            refresh.click(begin_update, inputs=None, outputs=lifecycle_outputs,
                          queue=False, show_progress="hidden", api_name=False).then(
                          refresh_data, inputs=None,
                          outputs=[*view_outputs, note_component, table, status],
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
    # 依目前測站及模型選項重新讀取儀表板資料。
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


# 定義圖表來源、測站、提前小時與網頁啟動參數。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["both", "8h", "single"], default="both",
                        help="both：同時比較兩種模型（預設）；8h：多步模型；single：單步模型")
    parser.add_argument("--horizon", type=int, choices=range(1, 9), default=1,
                        help="8h 模型的提前小時，預設 h1，每個目標時間只取一筆")
    parser.add_argument(
        "--station",
        default="木柵站",
        help="資料庫中的測站名稱，預設為木柵站",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=7,
        help="圖表初始顯示最後幾天，預設為 7 天",
    )
    parser.add_argument("--port", type=int, default=None, help="Gradio 連接埠，預設自動尋找可用埠（由 7860 起）")
    parser.add_argument("--no-browser", action="store_true", help="啟動服務但不自動開啟瀏覽器")
    return parser


def launch_dashboard(args, update_callback=None) -> int:
    """使用已解析的設定啟動，供 main.py 共用，避免重複解析命令列。"""
    show_prediction_visualization(
        station=args.station,
        initial_window_days=args.window_days,
        source=args.source,
        horizon=args.horizon,
        server_port=args.port,
        inbrowser=not args.no_browser,
        update_callback=update_callback,
    )
    return 0


# 解析命令列參數並啟動預測儀表板。
def main(argv=None, update_callback=None) -> int:
    return launch_dashboard(build_parser().parse_args(argv), update_callback=update_callback)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, SQLAlchemyError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        raise SystemExit(1)
