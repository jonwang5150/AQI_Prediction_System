# DL_AQI：逐時 AQI 預測與互動式圖表

本專案使用臺北市逐時空氣品質資料，透過兩種 PyTorch LSTM 模型預測 AQI，將觀測與預測儲存至 PostgreSQL，再以 Gradio 與 Plotly 顯示趨勢及誤差比較。

執行 `main.py` 會先載入既有歷史預測。按下「更新至最新」後，系統會接續爬取已發布的觀測、執行兩個模型，並在同一個頁面刷新圖表。v2 另提供以最新有效觀測為基準的未來 8 小時預測。

## 目錄

- [主要功能與流程](#主要功能與流程)
- [專案結構](#專案結構)
- [環境與安裝](#環境與安裝)
- [資料庫設定](#資料庫設定)
- [快速啟動與首次建置](#快速啟動與首次建置)
- [Gradio 介面與更新流程](#gradio-介面與更新流程)
- [爬蟲使用方式](#爬蟲使用方式)
- [模型訓練](#模型訓練)
- [歷史回測與未來預測](#歷史回測與未來預測)
- [直接使用現成資料](#直接使用現成資料)
- [評估指標與資料處理](#評估指標與資料處理)
- [常見問題](#常見問題)
- [目前範圍與維護](#目前範圍與維護)

## 主要功能與流程

| 功能 | 說明 |
|---|---|
| 觀測資料爬取 | 依測站與日期下載，每個下載批次最多 30 個日曆日 |
| 資料庫更新 | 驗證資料後批次 upsert，相同主鍵更新既有資料 |
| v1 單步預測 | 使用過去的污染物特徵預測下一小時 AQI |
| v2 多步預測 | 使用過去的 AQI 與污染物特徵，一次輸出未來 8 小時 AQI |
| 圖表切換 | v1、v2、三線比較、MAE／RMSE／Bias，按鈕切換呈現 |
| 更新至最新 | 接續下載新觀測、增量預測與失敗補算、刷新圖表 |
| 現成資料輸入 | 支援 DataFrame、爬蟲回傳列，以及 CSV／ODS／XLSX／XLS |

```mermaid
flowchart TD
    A[歷史 CSV] --> B[pipeline_v1 / pipeline_v2 訓練]
    B --> C[模型 checkpoint 與 scaler]
    D[臺北市 AQI 下載網站] --> E[crawl_to_db.py]
    E --> F[(PostgreSQL 觀測資料)]
    E --> G[記憶體中的爬蟲資料]
    F --> H[預測輸入與歷史視窗]
    G --> H
    C --> I[predict_1h.py / predict_8h.py]
    H --> I
    I --> J[(PostgreSQL 預測資料)]
    F --> K[真實值與預測對齊]
    J --> K
    K --> L[Gradio 趨勢圖與評估]
    M[main.py] --> L
    L -->|按下更新至最新| E
```

「更新」使用既有 checkpoint 推論，不重新訓練模型。

## 專案結構

```text
DL_AQI/
├── main.py                       # Gradio 主入口，串接爬取、預測與刷新
├── visualizations.py        # 圖表入口與顯示參數
├── crawl_to_db.py                 # 網站下載、解析、逐批寫入並回傳合併資料
├── update_db.py                   # 共用資料庫連線、欄位解析與交易寫入
├── db_models.py                   # 三張資料表的 ORM 定義與 upsert
├── config.py                      # 資料庫網址與寫入預設值
├── prediction_data_process.py     # 共用觀測查詢、資料來源選擇、時間整理與標準化
├── model_checkpoint.py            # 共用 checkpoint 載入、scaler 驗證與模型還原
├── predict_1h.py                # v1 指定期間逐時預測
├── predict_8h.py             # v2 逐時多步回測及最新基準預測
├── aqi_lstm_model.pth             # v1 checkpoint
├── aqi_lstm_8h_model.pth           # v2 checkpoint
├── pipeline_v1/
│   ├── __init__.py
│   ├── run_model.py              # v1 訓練入口
│   ├── data_cleaning.py          # CSV 清理
│   ├── data_split.py             # 時間切分、標準化與序列建立
│   ├── model.py                  # 單步 LSTM 與 checkpoint 儲存
│   ├── train.py                  # 訓練與評估
│   ├── predict.py                # 下一小時預測
│   └── visualization.py          # 訓練資料趨勢與特徵關係圖
├── pipeline_v2/
│   ├── __init__.py
│   ├── run_model_new.py          # v2 訓練入口
│   ├── data_cleaning_new.py      # CSV 清理
│   ├── data_split_new.py         # 多步序列與時間切分
│   ├── model_new.py              # 多步 LSTM 與 checkpoint 儲存
│   ├── train_new.py              # 訓練與逐 horizon 評估
│   └── predict_new.py            # 一次預測未來多小時
├── README.md                      # 專案操作、更新介面與維護說明
└── CODE_WORKFLOW.md               # 舊版下載腳本流程紀錄
```

實際圖表檔名是 `visualizations.py`。`CODE_WORKFLOW.md` 中提到的 `search_v2.py`、`serch.py` 不在目前專案中；現行下載入口為 `crawl_to_db.py`，操作以本 README 與目前程式為準。

### 共用邏輯與維護位置

| 調整項目 | 共用位置 |
|---|---|
| PostgreSQL driver、連線逾時與 engine 釋放 | `update_db.create_database_engine()`、`database_connection()` |
| 批次寫入、交易與預設 batch size | `update_db._write_rows()`；ORM upsert 在 `db_models.py` |
| checkpoint 讀取、scaler 驗證、模型還原 | `model_checkpoint.py` |
| 資料庫觀測欄位與時間篩選 | `prediction_data_process.read_database_observations()` |
| 記憶體資料／檔案／資料庫的來源選擇 | `prediction_data_process.resolve_observation_data()` |
| 預測區間、逐時時間列、前向補值與標準化 | `prediction_data_process.py` |
| 圖表啟動參數與 Gradio 入口 | `visualizations.build_parser()`、`launch_dashboard()` |

`predict_1h.py` 與 `predict_8h.py` 保留各自的序列推論與結果寫入流程，並沿用 `load_checkpoint()`、`load_database_data()`、`main(data=..., argv=...)` 介面。v2 仍要求觀測涵蓋輸入區間首尾；共用補值函式只產生模型輸入，不修改評估用的原始真實值。`main.py` 的增量進度判斷不與一般歷史回測混用。

## 環境與安裝

需要 Python、PostgreSQL，以及可連線至資料來源網站的網路。程式目前使用 CPU 訓練與推論，沒有實作 GPU 裝置切換。

以下命令在專案根目錄執行。請確定安裝套件與執行程式使用同一個 Python：

```powershell
python -c "import sys; print(sys.executable); print(sys.version)"
python -m pip install numpy pandas torch scikit-learn "SQLAlchemy>=2.0,<2.1" "psycopg[binary]>=3,<4" requests beautifulsoup4 odfpy plotly "gradio==6.5.1"
```

若要讀取 Excel 檔案，另安裝：

```powershell
python -m pip install openpyxl xlrd
```

| 套件 | 用途 |
|---|---|
| `numpy`、`pandas` | 時序資料處理與計算 |
| `torch` | LSTM 訓練、checkpoint 載入與推論 |
| `scikit-learn` | 訓練資料的 `StandardScaler` |
| `SQLAlchemy`、`psycopg` | PostgreSQL ORM、連線及交易 |
| `requests`、`beautifulsoup4` | 網站表單解析與下載 |
| `odfpy` | 解析來源網站回傳的 ODS |
| `openpyxl`、`xlrd` | 分別解析 XLSX、XLS |
| `plotly`、`gradio` | 互動圖表與操作介面 |

本機曾驗證的環境為 Python **3.13.9**、Gradio **6.5.1**、pandas **2.3.3**、Plotly **6.3.0**、PyTorch **2.11.0+cu126**。這是環境紀錄，不代表完整版本相容性測試或鎖定檔。

在目前這台電腦，實際使用的 Python 路徑為：

```powershell
& "C:\Users\jon\anaconda3.14\python.exe" -m pip install odfpy
& "C:\Users\jon\anaconda3.14\python.exe" main.py
```

資料夾名 `anaconda3.14` 不代表 Python 版本一定是 3.14。VS Code 的「Python: Select Interpreter」也應選擇同一環境。`main.py` 啟動時會印出執行環境路徑。

## 資料庫設定

修改 [config.py](config.py)，填入自己的連線資訊：

```python
class Settings:
    database_url = "postgresql+psycopg://YOUR_USER:YOUR_PASSWORD@localhost:5432/YOUR_DATABASE"
    batch_size = 1000
    dry_run = False

settings = Settings()
```

上述帳密是佔位文字。PostgreSQL 服務與指定的資料庫需先存在，使用者需有 `public` schema 的建表、查詢與寫入權限。程式會在寫入時建立尚不存在的資料表，但不會建立 PostgreSQL 資料庫，也不會自動遷移既有表格結構。

目前程式直接讀取 `config.py`，沒有自動載入 `.env`。分享專案時請勿附上真實帳密。

### 資料表

| ORM 類別 | PostgreSQL 資料表 | 主鍵 | 內容 |
|---|---|---|---|
| `AirQualityHourly` | `public.air_quality_hourly` | `station, observed_at` | 原始逐時 AQI、空氣品質指標及六項污染物 |
| `AirQualityPrediction` | `public.air_quality_predictions` | `station, predicted_at` | v1 預測與對應實際 AQI |
| `AirQualityPrediction8h` | `public.air_quality_predictions_8h` | `station, target_at, horizon_hours, model_version` | v2 各提前小時的預測、模型版本與寫入時間 |

v2 的 `model_version` 使用 checkpoint 檔案的 SHA-256。不同模型版本可並存，圖表按目標時間選取目前查詢條件下最新寫入的版本。v1 不保留多個模型版本，相同測站與目標時間會覆寫。

觀測時間與預測目標時間使用**台灣當地無時區時間**；v2 的 `created_at` 使用帶時區的 UTC 時間。v2 不儲存真實 AQI 副本，顯示時以測站與目標時間連接觀測表。

## 快速啟動與首次建置

### 已有模型、觀測與兩種歷史預測

```powershell
python main.py
```

依終端機顯示的本機網址開啟網頁。預設尋找從 `7860` 起的可用連接埠，且會自動開啟瀏覽器。

### 空資料庫的建置順序

1. 安裝套件並設定 PostgreSQL。
2. 準備兩個 checkpoint；若沒有，先依[模型訓練](#模型訓練)產生。
3. 匯入歷史觀測，包含第一筆預測需要的歷史視窗。
4. 分別產生 v1、v2 歷史預測。
5. 啟動 Gradio。

以下範例建立預設的 `2026-01-01` 至 `2026-07-31` 歷史圖表。爬蟲起始日向前多取資料，以涵蓋兩個模型的輸入視窗：

```powershell
python crawl_to_db.py --stations 木柵站 --start 2025-12-30 --end 2026-07-31 --no-dry-run
python predict_1h.py --station 木柵站
python predict_8h.py --station 木柵站
python main.py --station 木柵站
```

啟動 Gradio 不會自動補建歷史預測；預設 `--source both` 需要兩個模型都有可繪製資料。只有一種預測時，可先用 `--source single` 或 `--source 8h` 檢視。

## Gradio 介面與更新流程

### 啟動參數

`main.py` 與 `visualizations.py` 使用相同的顯示參數：

| 參數 | 預設 | 說明 |
|---|---|---|
| `--station` | `木柵站` | 資料庫中的測站名稱 |
| `--source` | `both` | `both`、`single` 或 `8h` |
| `--horizon` | `1` | v2 歷史比較使用的提前小時，範圍 1～8 |
| `--window-days` | `7` | 趨勢圖初始顯示的最近天數，不限制載入或評估期間 |
| `--port` | 自動選擇 | 指定 Gradio 連接埠 |
| `--no-browser` | 不啟用 | 啟動服務但不自動開啟瀏覽器 |

```powershell
python main.py --station 木柵站 --horizon 8 --window-days 14 --port 7860
python -m visualizations --source both --no-browser
```

`main.py` 呼叫 `visualizations.py` 啟動 Gradio，並提供更新功能。啟動時只顯示既有預測，不自動爬取或執行模型；單獨執行 `visualizations.py` 則是檢視介面。

### 四種圖表

- **v1 單步模型 vs 真實數據**：藍色預測線與橘色實際 AQI。
- **v2 8h 模型 vs 真實數據**：綠色預測線與橘色實際 AQI。
- **三線比較**：v1、v2 與真實 AQI。
- **評估比較**：MAE、RMSE、Bias 長條圖與評估明細。

每次呈現一張主要圖表。趨勢圖可拖曳、縮放，並切換 1 天、7 天、1 個月或全部期間。

### 「更新至最新」會做什麼

1. 讀取兩個 checkpoint、指定測站觀測表的最後時間，以及兩個模型各自已完成的預測進度。查詢觀測表最後時間時僅考慮不晚於當下整點的資料，不要求該列 AQI 非空。
2. 從觀測資料表最後時間的下一小時接續爬取至當下整點，沒有固定的 8/1 起點。
3. 網站按日下載；寫入前排除已處理時間、按鈕按下當時整點之後的資料，以及全空白的未發布列。
4. 合併既有觀測、必要歷史視窗與爬蟲回傳的資料，新下載部分不再從資料庫讀回。
5. 各模型從最後完成目標的下一小時補算至最新有效觀測；v2 另產生最新基準的未來 8 小時。沒有新增有效觀測且沒有待補算目標時，保留既有預測。
6. 寫入結果，重新載入同一個頁面的圖表與指標，不另開 Gradio 伺服器。

「更新至最新」按鈕靠左，固定為 144 × 36 px。更新期間按鈕變成「更新中…」並停用，舊圖表與切換按鈕暫時隱藏，顯示單一進度區。結束後恢復操作；若失敗，恢復原有圖表並顯示失敗步驟與可呈現的原因。進度百分比代表流程階段，不是精確的剩餘時間估計。

### 增量範圍與重試

爬取起點與預測起點分開計算：

| 用途 | 起點／判斷方式 |
|---|---|
| 爬取新觀測 | 觀測表最後時間 + 1 小時 |
| v1 補算 | 不晚於既有觀測截止時間的最後預測目標 + 1 小時 |
| v2 補算 | 不晚於既有觀測截止時間、同一模型版本具備完整 h1～h8 的最後目標 + 1 小時 |
| 預測截止 | 合併資料中最新有有效 AQI 的觀測時間 |
| 模型輸入 | 最早待補算目標之前的必要歷史視窗，加上待補算及新下載的觀測 |

例如觀測表與兩個模型皆已完成至 `2026-10-11 13:00`，現在為 `15:40`：爬蟲更新範圍為 `14:00`～`15:00`，模型只處理新增有效觀測。網站按整天提供下載，因此回應中仍可能包含當天較早的小時，但程式會篩除，不會重複寫入。若網站只發布至 `14:00`，本次預測只推進至該時間，下一次再從觀測表最後時間接續。

爬蟲每個下載批次獨立提交，兩個模型也各自寫入，整個更新不是單一交易。中途失敗時，成功寫入的資料會保留；重試會接續爬取，並依各模型最後完成的目標補算。v2 以同一模型版本具備 h1～h8 的目標作為完成進度，避免誤把未來曲線當成已完成的歷史回測。資料庫只讀取待預測區間及必要歷史視窗；24 小時窗口的兩個模型通常只需讀取最後 31 小時既有觀測。

這 31 小時是模型輸入所需的資料庫讀取，不是重新爬取歷史資料。若前次模型落後，才會另讀尚未完成的期間。當某模型沒有可用的完成進度時，增量入口以觀測表最後時間的下一小時為起點；首次建立歷史預測仍需依「快速啟動與首次建置」操作。更早的缺漏或觀測修訂不會自動回溯爬取。

### 未來 8 小時的意義

若最新有效觀測是 `2026-10-11 13:00`，v2 最新一輪會預測 `14:00` 至 `21:00`。這八個點依次為 h1～h8，以綠色虛線顯示，未來期間另有淡綠底色。

`--horizon 1` 控制的是歷史比較曲線，不會把最新一輪未來八小時全部變成 h1。v2 一次輸出八個值，不是遞迴執行單步模型八次。

網站可能延遲發布；未來預測基準是**最新有效觀測**，不一定是電腦現在時間。v1 的更新流程只產生至最新觀測時間的歷史預測，不額外延伸未來一小時。

## 爬蟲使用方式

資料來源：[臺北市環境品質資訊網 AQI 小時資料下載頁](https://www.tldep.gov.taipei/Public/DownLoad/AqiHour.aspx)。程式解析網站表單與測站選項，下載的試算表直接在記憶體解析，不自動儲存中間檔案。

```powershell
# 列出網站測站
python crawl_to_db.py --list-stations

# 指定測站與日期，寫入資料庫
python crawl_to_db.py --stations 木柵站 --start 2026-08-01 --end 2026-08-31 --no-dry-run

# 多個測站或全部測站
python crawl_to_db.py --stations "木柵,大安" --start 2026-08-01 --end 2026-08-02
python crawl_to_db.py --stations all --start 2026-08-01 --end 2026-08-02

# 僅下載、解析及驗證，不連線寫入資料庫
python crawl_to_db.py --stations 木柵站 --start 2026-08-01 --end 2026-08-02 --dry-run
```

| 參數 | 預設／行為 |
|---|---|
| `--stations` | 未提供則互動詢問；支援逗號分隔、`all`、`*`、`全部`、`全選` |
| `--start`、`--end` | 未提供則互動詢問；支援 `YYYY-MM-DD` 或 `YYYY/MM/DD`，兩端日期皆包含 |
| `--delay` | 下載批次間隔，預設 1 秒 |
| `--timeout` | HTTP 逾時，預設 60 秒 |
| `--batch-size` | 資料庫寫入批次大小，預設取 `config.py` |
| `--dry-run` | 只驗證，不寫入；預設取 `config.py` |
| `--no-dry-run` | 強制執行寫入 |

「木柵站」會對應網站的「木柵」選項。`main.py` 更新時另以 `station_names` 將下載列統一回資料庫使用的名稱；單獨使用爬蟲而未指定此映射時，保留下載資料本身的測站名稱。

程式呼叫 `crawl_to_db.main(argv)` 會回傳合併後的 `list[dict]`，可重複交給兩個模型。`--dry-run` 也會回傳資料；`--list-stations` 回傳空清單。

## 模型訓練

### 資料格式

訓練入口預設讀取根目錄的 `20250101-20251231.csv`，**目前專案未附此檔案**，請自行準備，或使用 `--data` 指定。訓練資料應先篩選為單一測站；訓練清理函式不會替你按測站分組。

CSV 使用 UTF-8／UTF-8 BOM，欄位範例如下。以下只有格式示意，不足以訓練：

```csv
測站,日期,AQI,空氣品質指標,O3,PM2.5,PM10,CO,SO2,NO2
木柵站,2025/01/01 00:00,40,良好,20.0,10.0,18.0,0.30,1.0,12.0
木柵站,2025/01/01 01:00,42,良好,21.0,11.0,19.0,0.31,1.1,13.0
```

訓練需要「日期」、「AQI」及六個污染物欄位；「測站」、「空氣品質指標」不參與模型輸入。爬蟲解析下載檔時則要求網站的完整十個欄位。

### v1 與 v2 比較

| 項目 | v1 單步模型 | v2 多步模型 |
|---|---|---|
| 輸入特徵 | O3、PM2.5、PM10、CO、SO2、NO2 | AQI 加上 v1 的六項污染物 |
| 預設輸入窗口 | 24 小時 | 24 小時 |
| 輸出 | 下一小時 AQI | 未來 1～8 小時 AQI |
| 網路 | 單層 LSTM，hidden size 64；Linear 32 → ReLU → 輸出層 | 同類架構，輸出層為 8 個值 |
| 預設訓練 | 50 epochs、Adam、learning rate 0.001、MSE loss | 相同 |
| 資料切分 | 依時間前 80% 訓練、後 20% 測試 | 相同 |
| 訓練 batch size | 32 | 32 |
| 儲存檔 | `aqi_lstm_model.pth` | `aqi_lstm_8h_model.pth` |

StandardScaler 只在訓練集擬合；測試集與後續推論使用相同的縮放參數。AQI 輸出維持原始尺度，v2 另保留未標準化的目標欄位。

```powershell
python -m pipeline_v1.run_model --data 20250101-20251231.csv --epochs 50 --window-size 24 --no-visualization
python -m pipeline_v2.run_model_new --data 20250101-20251231.csv --epochs 50 --window-size 24 --forecast-horizon 8
```

兩者都可使用 `--model` 指定輸出 checkpoint。訓練預設路徑相對於目前工作目錄，因此請從專案根目錄執行，讓 `main.py` 找得到輸出檔案。v1 未加 `--no-visualization` 時會開啟訓練資料的 Plotly 圖表。

checkpoint 包含權重、模型設定、特徵順序、窗口大小與 scaler；v2 另有 `forecast_horizon`。雖然 v2 訓練入口可設定其他輸出長度，目前 `predict_8h.py` 與資料表限制為 **8 小時**。

## 歷史回測與未來預測

兩支預測程式預設區間皆為 `[2026-01-01 00:00, 2026-08-01 00:00)`，即包含開始、不包含結束。這與爬蟲包含結束日期的規則不同。

### v1：逐時單步預測

```powershell
python predict_1h.py --station 木柵站 --start 2026-08-01 --end 2026-08-12
```

可用參數：`--station`、`--model`、`--start`、`--end`、`--data`。未提供現成資料就查詢資料庫，完成後寫入 v1 預測表。**v1 目前沒有 `--dry-run` 參數。**

### v2：逐時歷史回測

```powershell
python predict_8h.py --station 木柵站 --start 2026-08-01 --end 2026-08-12
python predict_8h.py --station 木柵站 --start 2026-08-01 --end 2026-08-12 --dry-run
```

除上述共用參數外，v2 有 `--batch-size`（預設 256）、`--dry-run`、`--latest-origin`。一般回測模式會為每個目標小時保留 h1～h8；例如一天有 `24 × 8 = 192` 筆預測。

`horizon_hours` 是提前幾小時預測；預測基準時間可以由下式推得：

```text
基準時間 = target_at - horizon_hours 小時
```

### v2：以最新觀測預測未來

假設資料確實已包含 `2026-10-11 13:00` 的觀測：

```powershell
python predict_8h.py --station 木柵站 --start 2026-08-01 --latest-origin "2026-10-11 13:00" --end "2026-10-11 22:00"
```

此模式的 `--end` 必須等於 `latest-origin + 9 小時`，因為結束時間不包含；最後預測目標是 `latest-origin + 8 小時`。只使用截至最新基準的觀測，不以虛構的未來輸入補滿窗口。越遠的未來目標尚未具備全部八種歷史基準，因此不要求每個未來目標都有 h1～h8 八筆。

### 歷史視窗需求

窗口為 24 小時，目標開始為 `2026-08-01 00:00` 時：

| 模型 | 最早所需觀測 | 原因 |
|---|---|---|
| v1 | `2026-07-31 00:00` | 目標前 24 小時 |
| v2 | `2026-07-30 17:00` | 第一個目標的 h8 基準在開始前 8 小時，再向前取 24 小時窗口 |

v2 一般回測模式需要觀測到 `end - 2 小時`；最新基準模式需要觀測到 `latest-origin`。v2 會拒絕把不足的首尾觀測範圍外推補齊。

## 直接使用現成資料

### 從檔案讀取

```powershell
python predict_1h.py --data downloaded.csv
python predict_8h.py --data batch1.ods batch2.ods --dry-run
```

`--data` 可指定多個檔案，支援 CSV、ODS、XLSX、XLS。資料必須覆蓋所選預測期間及歷史視窗；重疊批次造成的重複時間須先整理。

### 從爬蟲直接交給模型

```python
import crawl_to_db
import predict_1h
import predict_8h

station = "木柵站"
rows = crawl_to_db.main(
    ["--stations", station,
     "--start", "2025-12-30", "--end", "2026-07-31", "--no-dry-run"],
    station_names={"木柵": station},
)

predict_1h.main(data=rows, argv=["--station", station])
predict_8h.main(data=rows, argv=["--station", station])
```

`data` 可為 pandas DataFrame 或爬蟲列的 iterable。若兩個模型要共用資料，使用清單或 DataFrame，避免共用已耗盡的 generator。程式化呼叫請傳入 `argv`，避免讀取呼叫端自己的命令列參數。

`prediction_data_process.py` 負責以下對應與檢查：

| 爬蟲／資料庫欄位 | 模型資料欄位 |
|---|---|
| `station` | `測站` |
| `observed_at` | `日期` |
| `aqi` | `AQI` |
| `o3`、`pm25`、`pm10` | `O3`、`PM2.5`、`PM10` |
| `co`、`so2`、`no2` | `CO`、`SO2`、`NO2` |

它會篩選指定測站與時間、排序、轉換數值，並拒絕無效日期、非整點及重複時間。缺少測站欄位時，視為呼叫端已選好測站。

提供 `data` 或 `--data` 時不呼叫模型的 `load_database_data()`；兩者不可同時提供。提供空資料或無效資料會報錯，不會偷偷改查資料庫。跳過觀測讀取不等於跳過結果寫入：v1 仍寫入，v2 需加 `--dry-run` 才不寫入。

## 評估指標與資料處理

令誤差 `e = 預測 AQI − 實際 AQI`：

| 指標 | 計算 | 解讀 |
|---|---|---|
| MAE | `mean(abs(e))` | 平均絕對誤差，越低越好 |
| RMSE | `sqrt(mean(e²))` | 大誤差影響較高，越低越好 |
| Bias | `mean(e)` | 正值代表平均高估，負值代表平均低估 |

Gradio 使用所有已載入模型**共同具有有限預測值與真實值的目標時間**計算指標，不補齊缺失真實值。只有一個模型時使用該模型的有效時間點。

- 評估明細包括有效筆數、未納入筆數、評估開始與結束。
- 未來預測尚無真實值，不納入指標。
- 無共同有效時間點時，顯示 N/A。
- 指標使用載入資料的全期間；縮放趨勢圖或改變初始顯示天數不會重算。

**訓練與推論的補值方式不同：**訓練 CSV 清理使用線性插值；根目錄兩支預測程式使用前向補值，避免用未來觀測填補過去的輸入。訓練目前在時間切分前做插值，因此不能把訓練入口輸出的測試分數視為已完全排除前處理資訊洩漏的嚴格驗證結果。

## 常見問題

### 更新時出現缺少 `odfpy`

先看 `main.py` 印出的 Python 路徑，再用**相同執行檔**安裝：

```powershell
python -c "import sys; print(sys.executable)"
python -m pip install odfpy
```

若同時有多個 Anaconda／Python，請使用完整路徑。套件裝在另一個環境，不會解決目前程式的缺少套件錯誤。

### 找不到測站，或圖表沒有資料

先執行 `python crawl_to_db.py --list-stations`。網站可能使用「木柵」，資料庫則使用「木柵站」。爬蟲可對應名稱，但預測與圖表查詢要使用資料庫實際名稱。

確認兩張預測表都已產生該測站的資料。若只有其中一種，使用對應的 `--source`。

### 觀測範圍不足／特徵補值後仍含空值

確認提供了預測開始之前的歷史視窗，以及所需的最後觀測時間。起始窗口的缺值沒有更早資料可供前向補值，需補齊原始觀測，不能只下載目標期間。

### PostgreSQL 寫入失敗

檢查 `config.py`、PostgreSQL 服務、資料庫名稱、`public` schema 權限及資料表結構。程式只建立缺少的表，不會自動修改舊表。錯誤訊息會標示下載、解析或寫入階段，終端機保留原因鏈中的位置資訊。

### 介面修改後沒有變化

終端機按 `Ctrl+C` 停止原本的 `main.py`，重新執行，再刷新網頁。只刷新瀏覽器不會重新載入 Python 原始碼。多個服務同時運行時，也要確認開啟的是新服務輸出的網址及連接埠。

### `To create a public link, set share=True in launch()`

這是 Gradio 的公開連結提示，不是錯誤。目前服務綁定 `127.0.0.1` 且 `share=False`，用本機網址即可。

### 最新資料仍比現在早

來源網站可能尚未發布目前小時的資料。狀態訊息會顯示真正取得的最新 AQI 時間；未來 8 小時以該時間為基準，而非強制從現在起算。

## 目前範圍與維護

- 更新起點依觀測表與兩個模型的完成進度向前推進，不固定在任何日期。
- 只有新增或待補算區間會執行預測；圖表仍會載入既有歷史預測供查看。更換模型不會自動重算全部歷史，需要時請手動指定期間執行預測程式。
- 更新功能不回溯爬取既有時間的修訂，也不自動修補最後觀測之前的缺漏。需要時手動重爬該期間。
- `main.py` 使用根目錄兩個固定檔名的 checkpoint；獨立預測與訓練入口才提供 `--model`。
- 未提供排程更新、自動重新訓練、登入驗證或正式對外部署設定。
- checkpoint 載入可能使用 pickle 相容模式，應使用自己訓練或可信任來源的檔案。
- 目前未提供完整依賴鎖定檔；RAG 查詢與介面整合可用 `python -m unittest test_rag_query -v` 測試。

修改後可先用以下命令核對介面參數及爬蟲解析：

```powershell
python main.py --help
python predict_1h.py --help
python predict_8h.py --help
python crawl_to_db.py --stations 木柵站 --start 2026-08-12 --end 2026-08-12 --dry-run
```

再視需求使用 v2 的 `--dry-run` 驗證推論，最後檢查四種圖表、單一進度面板，以及更新失敗後的恢復行為。上述需要實際資料的操作，仍取決於來源網站與本機資料庫是否可用。

## RAG 知識庫查詢

每日摘要包含實際 AQI 最高／最低值、所有並列的測站與時間、有效觀測筆數與當日資料截止時間。極值直接使用所有測站的有效觀測，不限是否有預測配對；平均與誤差指標仍僅使用有效配對。只有觀測的日期也會建立摘要，無配對的指標標示無法計算。執行 `python build_pgvector.py` 可依原有固定 ID 更新文件及向量，無須刪表；資料截止時間不代表全日每小時皆已完整。

啟動 `python main.py` 後，按圖表功能列中的「RAG 查詢」切換到查詢畫面，輸入完整問題並按「送出查詢」。回答下方的「查看參考摘要」列出實際檢索的文件；每次提問獨立處理，不保留對話上下文。按同一列的圖表或評估按鈕可切回對應畫面，查詢內容與回答會保留。

查詢使用 `aqi_knowledge_base` 集合，只檢索每日與每月評估摘要，不會重新匯入文件。請先啟動 Ollama、準備設定的 embedding／聊天模型，並成功執行 `python build_pgvector.py`。設定沿用 `config.py` 的 `ollama_base_url`、`ollama_embedding_model`、`ollama_chat_model` 與 `top_k`；雲端聊天模型需要有效授權，問題與檢索摘要會送到該模型。

摘要合併所有測站，使用最新寫入的 8h 預測，不套用圖表的測站或固定 h 篩選。明確指定單一年月（例如 `2026 年 8 月`、`2026-08`）時會篩選月摘要；指定年月日時會篩選當日摘要。月內每日問題會篩選該月的每日文件，但仍只取 top_k 筆；相對日期與複雜日期區間尚未支援精確解析。語意檢索僅取部分摘要，不能當作全期間排序或精確 SQL 統計。月摘要代表有效配對樣本，不保證整月資料完整。回答中的 `[1]` 等編號對應參考摘要。模型無法連線時會顯示錯誤並保留已檢索的摘要供查看。
