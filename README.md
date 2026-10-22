# DL_AQI：木柵 AQI 預測、視覺化與智能助理

本專案以臺北市逐時空氣品質資料為基礎，使用兩個 PyTorch LSTM 模型預測 AQI，將觀測與預測結果寫入 PostgreSQL，並以 Gradio、Plotly 提供互動式圖表。系統也整合 LangChain Agent、PGVector RAG、唯讀 SQL 查詢、木柵天氣預報及 Matplotlib 自動製圖。

目前智能助理與 RAG 的服務範圍固定為「木柵站／木柵地區」；圖表及預測命令雖可傳入其他測站名稱，但不會改變智能助理的固定服務範圍。

## 主要功能

| 功能 | 說明 |
|---|---|
| AQI 資料爬取 | 從臺北市政府環保局下載逐時資料，單次下載批次最多 30 個日曆日 |
| PostgreSQL 儲存 | 觀測、v1 預測及 v2 預測皆採主鍵 upsert，重跑相同期間不會新增相同主鍵 |
| v1 單步模型 | 使用前 24 小時的 6 種污染物特徵，預測下一小時 AQI |
| v2 多步模型 | 使用前 24 小時的 AQI 與 6 種污染物特徵，一次輸出未來 1～8 小時 AQI |
| 歷史回測 | 可指定半開區間 `[start, end)`，將結果寫回 PostgreSQL |
| 增量更新 | Gradio 的「更新至最新」會下載新觀測、從各模型最後進度後接續預測、更新向量庫並刷新圖表 |
| 互動式圖表 | 顯示 v1、v2、真實 AQI、最新未來 8 小時預測，以及 MAE／RMSE／Bias |
| RAG 知識庫 | 將木柵站 2026 年每日及每月預測評估摘要寫入 PGVector |
| AQI 智能助理 | 依問題選用 RAG、唯讀 SQL、天氣或 Matplotlib 工具，並保留單次網頁 session 對話 |
| 木柵天氣 | 直接呼叫 Open-Meteo，不使用模型生成氣象數值，提供目前天氣及最多 7 天預報 |
| 自動製圖 | Agent 將已取得的資料交給 Matplotlib 工具，產生 Python 程式與 PNG 圖檔 |

## 近期程式變更

- 圖表模組已整理至 `prediction_visualization/`，資料載入、評估與 UI 分開維護。
- v1、v2 訓練資料來源已由固定 CSV 改為依測站與日期直接查詢 PostgreSQL。
- v2 模組檔名統一使用 `_v2`，checkpoint 仍為 `aqi_lstm_8h_model.pth`。
- 訓練日期包含結束日全天；兩個 pipeline 依時間切分資料，僅以訓練集擬合 scaler，並以前段歷史資料銜接測試窗口。
- v2 將標準化的 AQI 輸入與原尺度標籤分開保存，評估時逐一輸出 h1～h8 的 MAE／RMSE。
- Gradio 已整合可連續追問的木柵 AQI 智能助理，並能顯示引用摘要與 Agent 產生的圖片。
- Agent 工具改為外部注入模型；SQL、RAG、Matplotlib 各自使用獨立模型實例，天氣工具完全不綁定模型。
- 原本的通用搜尋功能已不在目前專案，改為固定木柵座標的 Open-Meteo 天氣工具。
- RAG 改用 `KnowledgeBaseTool` 封裝，加入木柵 metadata 過濾、完整逐日序列模式及來源引用。
- 向量文件 ID 已納入 `source + station + period`，同一份摘要可穩定更新；舊規則留下的資料仍須以 SQL 清除後重建。
- `main.py` 的更新流程現在會在預測完成後呼叫 `build_knowledge_base()`，再重新載入圖表。

## 系統流程

```mermaid
flowchart TD
    A[臺北市逐時 AQI 下載網站] --> B[crawl_to_db.py]
    B --> C[(air_quality_hourly)]
    C --> D[pipeline_v1 / pipeline_v2 訓練]
    D --> E[模型 checkpoint 與 scaler]
    C --> F[predict_1h.py]
    C --> G[predict_8h.py]
    E --> F
    E --> G
    F --> H[(air_quality_predictions)]
    G --> I[(air_quality_predictions_8h)]
    C --> J[build_pgvector.py]
    I --> J
    J --> K[(PGVector aqi_knowledge_base)]
    C --> L[Gradio / Plotly]
    H --> L
    I --> L
    K --> M[AQI 智能助理]
    C --> M
    H --> M
    I --> M
    N[Open-Meteo] --> M
    M --> O[回答、參考摘要、Matplotlib 圖表]
```

按下 `main.py` 網頁中的「更新至最新」時，流程如下：

1. 載入 v1、v2 checkpoint，判斷模型所需的歷史窗口。
2. 讀取資料庫中最後觀測時間與兩個模型的預測進度。
3. 從最後觀測的下一小時開始下載已發布資料，不把未來空白列當成觀測。
4. 從 v1、v2 各自最後完成的時間之後接續產生目標時段；不會掃描並補齊更早的任意缺洞。
5. v2 以最新有效觀測為基準，額外產生未來 h1～h8。
6. 更新 `aqi_knowledge_base` 向量知識庫。
7. 重新載入圖表與共同有效時間點的評估指標。

更新工作有程序內鎖定；同一個 Python 程序不會同時執行兩次更新。

## 專案結構

```text
DL_AQI/
├─ main.py                              # 完整 Gradio 入口：圖表、更新、智能助理
├─ config.py                            # .env 載入與共用設定
├─ db_models.py                         # SQLAlchemy ORM 與 PostgreSQL upsert
├─ update_db.py                         # 驗證、建表及批次寫入
├─ crawl_to_db.py                       # 臺北市逐時 AQI 爬蟲
├─ prediction_data_process.py           # 兩種推論流程共用的資料處理
├─ model_checkpoint.py                  # checkpoint 與 scaler 驗證／還原
├─ predict_1h.py                        # v1 歷史逐時預測
├─ predict_8h.py                        # v2 h1～h8 回測及未來預測
├─ build_pgvector.py                    # 建立／更新木柵 2026 年評估摘要向量庫
├─ aqi_lstm_model.pth                   # v1 checkpoint
├─ aqi_lstm_8h_model.pth                # v2 checkpoint
├─ pipeline_v1/                         # v1 清理、切分、模型、訓練與評估
│  ├─ run_model.py                      # 串接清理、圖表、訓練、評估及存檔
│  ├─ data_cleaning.py                  # 讀取 DB、排序及污染物線性補值
│  ├─ data_split.py                     # 時間切分、標準化及單步序列
│  ├─ model.py                          # LSTMModel 與 checkpoint 儲存
│  ├─ train.py                          # 訓練及測試集 MAE／RMSE
│  ├─ predict.py                        # 最新測試窗口的下一小時預測
│  └─ visualization.py                  # AQI 趨勢及特徵散佈矩陣
├─ pipeline_v2/                         # v2 清理、切分、模型、訓練與評估
│  ├─ run_model_v2.py                   # 串接多步訓練、評估、預測及存檔
│  ├─ data_cleaning_v2.py               # 讀取 DB、欄位檢查及 AQI／污染物補值
│  ├─ data_split_v2.py                  # 多步序列及原尺度 AQI 標籤
│  ├─ model_v2.py                       # MultiHorizonLSTM 與 checkpoint 儲存
│  ├─ train_v2.py                       # 訓練及逐 horizon MAE／RMSE
│  └─ predict_v2.py                     # 最新連續窗口的多步預測
├─ prediction_visualization/
│  ├─ load_prediction_data.py           # 載入及挑選最新預測版本
│  ├─ evaluate_prediction_data.py       # 共同樣本 MAE／RMSE／Bias
│  └─ visualizations.py                 # Plotly 圖表與 Gradio UI
├─ agent/
│  ├─ agent.py                          # Agent 規則、模型與工具編排
│  ├─ rag_tool.py                       # PGVector 檢索及完整逐日序列查詢
│  ├─ sql_tool.py                       # 白名單、唯讀 SQL 查詢工具
│  ├─ weather_tool.py                   # 固定木柵 Open-Meteo 工具
│  ├─ matplotlib_tool.py                # 產生並執行 Matplotlib 程式
│  └─ chat_prompt.py                    # LangChain prompt 組裝
└─ charts/                              # Agent 產生的 .py 與 .png 圖表
```

## 環境需求與安裝

- Python 3.10 以上
- PostgreSQL
- PostgreSQL `vector` extension（RAG 使用）
- 本機 Ollama 與 `nomic-embed-text`（預設用於建立及查詢 embedding）
- Ollama Cloud API 金鑰（目前 Agent 模型在 `agent/agent.py` 中連至 `https://ollama.com`）

建議在專案使用的同一個 Python 環境安裝：

```powershell
python -m pip install `
  numpy pandas torch scikit-learn `
  "SQLAlchemy>=2.0,<2.1" "psycopg[binary]>=3,<4" python-dotenv `
  requests beautifulsoup4 odfpy openpyxl xlrd `
  plotly "gradio==6.5.1" matplotlib `
  langchain langchain-core langchain-community `
  langchain-ollama langchain-postgres ollama httpx pydantic
```

確認目前使用的 Python：

```powershell
python -c "import sys; print(sys.executable); print(sys.version)"
```

### Ollama embedding 設定

啟動本機 Ollama 並下載預設 embedding 模型：

```powershell
ollama serve
ollama pull nomic-embed-text
ollama list
```

`build_pgvector.py` 與 `agent/rag_tool.py` 預設連到 `http://localhost:11434`。安裝 Ollama 不代表服務一定正在執行；若出現 `Failed to connect to Ollama`，先確認 `ollama serve` 及 11434 連接埠。

## 環境變數

在專案根目錄建立 `.env`。不要提交真實密碼或 API 金鑰：

```dotenv
DATABASE_URL=postgresql+psycopg://YOUR_USER:YOUR_PASSWORD@localhost:5432/YOUR_DATABASE

# 本機 embedding 服務；也可使用舊名稱 OLLAMA_BASE_URL
LLM_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
TOP_K=4

# 爬蟲與一般批次寫入
BATCH_SIZE=1000
DRY_RUN=false

# Agent 使用 Ollama Cloud 模型時需要
OLLAMA_API_KEY=YOUR_OLLAMA_API_KEY

# 可選：讓唯讀 SQL Agent 使用權限更小的 PostgreSQL 帳號
SQL_AGENT_DATABASE_URL=postgresql+psycopg://READ_ONLY_USER:PASSWORD@localhost:5432/YOUR_DATABASE
```

目前實際使用情形：

| 變數 | 使用位置 | 預設值／說明 |
|---|---|---|
| `DATABASE_URL` | 所有 PostgreSQL 功能 | 建議一定明確設定 |
| `LLM_BASE_URL` | embedding 建立與 RAG 檢索 | `http://localhost:11434` |
| `OLLAMA_BASE_URL` | `LLM_BASE_URL` 的相容舊名稱 | 僅在未設定 `LLM_BASE_URL` 時採用 |
| `EMBEDDING_MODEL` | embedding 模型 | `nomic-embed-text` |
| `OLLAMA_EMBED_MODEL` | embedding 模型相容舊名稱 | 僅在未設定 `EMBEDDING_MODEL` 時採用 |
| `TOP_K` | 一般 RAG 相似檢索筆數 | `4`，程式限制在 1～20 |
| `BATCH_SIZE` | 爬蟲寫入批次 | `1000` |
| `DRY_RUN` | 爬蟲預設是否只驗證 | `false` |
| `SQL_AGENT_DATABASE_URL` | SQL 工具 | 未設定時使用 `DATABASE_URL` |
| `OLLAMA_API_KEY` | Ollama Cloud | 由 Ollama client 讀取 |

`EMBEDDING_DIM`、`CHUNK_SIZE`、`CHUNK_OVERLAP`、`SCORE_THRESHOLD`、`AUTO_SEED`、`SEED_DATA_DIR` 目前會由 `config.py` 載入，但現有主流程沒有使用。

Agent 的模型目前不是由 `.env` 選擇：主 Agent、SQL 與 Matplotlib 使用 `gpt-oss:120b-cloud`，RAG 回答使用 `nemotron-3-ultra:cloud`，都在 `agent/agent.py` 中建立，再注入各工具。

## 資料庫結構

資料表會在首次寫入時建立，但 PostgreSQL 資料庫本身必須先存在。

| 資料表 | 主鍵 | 內容 |
|---|---|---|
| `public.air_quality_hourly` | `station, observed_at` | AQI、空氣品質指標及 6 種污染物觀測 |
| `public.air_quality_predictions` | `station, predicted_at` | v1 預測與當時保留的實際 AQI |
| `public.air_quality_predictions_8h` | `station, target_at, horizon_hours, model_version` | v2 h1～h8 預測、模型 SHA-256 版本及建立時間 |
| `public.langchain_pg_collection` | PGVector 管理 | 向量 collection 資訊 |
| `public.langchain_pg_embedding` | PGVector 管理 | 文件、metadata 與 embedding |

觀測時間、v1 的 `predicted_at` 及 v2 的 `target_at` 都使用臺灣當地「無時區」時間；v2 的 `created_at` 是帶時區時間。`predict_1h.py`、`predict_8h.py` 的目標區間採 `[start, end)`，包含開始、不包含結束；訓練 pipeline 的 `--start`、`--end` 則以日期指定，包含結束日全天。

## 快速啟動

若資料庫、checkpoint、Ollama embedding 與向量庫皆已準備好：

```powershell
python main.py
```

常用選項：

```powershell
python main.py --station 木柵站 --source both --horizon 1 --window-days 7 --port 7860
python main.py --station 木柵站 --no-browser
```

| 參數 | 預設 | 說明 |
|---|---|---|
| `--station` | `木柵站` | 圖表與更新流程使用的資料庫測站 |
| `--source` | `both` | `both`、`single` 或 `8h` |
| `--horizon` | `1` | v2 歷史曲線顯示 h1～h8 中的哪一條 |
| `--window-days` | `7` | 趨勢圖初始顯示天數 |
| `--port` | 自動 | Gradio 連接埠，未指定時由 7860 起找可用埠 |
| `--no-browser` | 關閉 | 啟動服務但不自動開瀏覽器 |

也可以只啟動既有資料的圖表，不啟用「更新至最新」按鈕：

```powershell
python -m prediction_visualization.visualizations --source both --horizon 1 --no-browser
```

### Gradio 畫面

- `v1｜單步模型 vs 真實數據`
- `v2｜8h 模型 vs 真實數據`
- `三線比較｜v1、v2、真實數據`
- `評估比較｜MAE / RMSE / Bias`
- `智能助理`

歷史 v2 曲線只顯示指定的 `--horizon`；按下更新後，最新一輪未來 h1～h8 會以綠色虛線顯示，沒有真實值，因此不納入誤差評估。若同一目標時間存在多個 v2 模型版本，圖表依 `created_at` 取最新寫入者。

MAE、RMSE 與 Bias 只使用所有已載入模型共同具有有限預測與真實值的時間點，不補造真實值，且不隨前端縮放範圍重新計算。

## 爬取觀測資料

資料來源：<https://www.tldep.gov.taipei/Public/DownLoad/AqiHour.aspx>

```powershell
# 查看網站目前提供的測站
python crawl_to_db.py --list-stations

# 匯入木柵資料
python crawl_to_db.py --stations 木柵站 --start 2026-08-01 --end 2026-08-31 --no-dry-run

# 多測站或全部測站
python crawl_to_db.py --stations "木柵,萬華" --start 2026-08-01 --end 2026-08-02 --no-dry-run
python crawl_to_db.py --stations all --start 2026-08-01 --end 2026-08-02 --no-dry-run

# 只下載、解析與驗證，不寫入資料庫
python crawl_to_db.py --stations 木柵站 --start 2026-08-01 --end 2026-08-02 --dry-run
```

重要行為：

- 日期起訖皆包含，超過 30 天時自動切成多個下載批次。
- 每批重新取得 ASP.NET 表單狀態，解析下載的試算表後直接寫入，不保留中間檔。
- HTTP GET 有重試與退避；連線仍驗證 CA 與主機名稱。
- 每個下載批次各自提交；後續批次失敗時，先前成功批次會保留。
- 同一 `station, observed_at` 會更新為最後讀到的內容，包括 `NULL`。
- `main()` 會回傳解析後的 `list[dict]`，可直接傳給預測程式。

## 模型訓練

兩個 pipeline 都透過 `config.py` 的 `DATABASE_URL`，從 `public.air_quality_hourly` 讀取指定測站與期間的觀測。執行前須先匯入資料；訓練入口不會自行爬取資料，也不接受 `--data` 檔案參數。

### v1／v2 比較

| 項目 | `pipeline_v1` | `pipeline_v2` |
|---|---|---|
| 執行入口 | `pipeline_v1.run_model` | `pipeline_v2.run_model_v2` |
| 模型類別 | `LSTMModel` | `MultiHorizonLSTM` |
| 輸入特徵順序 | `O3, PM2.5, PM10, CO, SO2, NO2` | `AQI, O3, PM2.5, PM10, CO, SO2, NO2` |
| 預設輸入序列形狀 | `(N, 24, 6)` | `(N, 24, 7)` |
| 預設標籤形狀 | `(N,)` | `(N, 8)` |
| 預測目標 | 下一小時 AQI | 一次輸出未來 h1～h8 AQI |
| 模型結構 | 單層 LSTM（hidden size 64）→ Linear（32）→ ReLU → Linear（1） | 單層 LSTM（hidden size 64）→ Linear（32）→ ReLU → Linear（預測步數） |
| 測試評估 | 單步 MAE／RMSE | 各 horizon 的 MAE／RMSE，以及各 horizon 指標的平均 |
| 預設 checkpoint | `aqi_lstm_model.pth` | `aqi_lstm_8h_model.pth` |

`N` 為成功建立的序列數；h1 表示輸入窗口最後一小時之後的第一小時。兩個模型都取 LSTM 最後一個時間步的輸出送入全連接層，標籤及模型預測皆保留 AQI 原尺度。v2 直接輸出多個時點，不會將前一個預測遞迴當成下一步輸入。

### 資料清理與序列建立

1. **讀取及排序**：依測站與日期查詢資料庫，按時間升冪排列，移除不使用的測站及空氣品質指標欄位。`--start`、`--end` 必須是有效的無時區日期，且開始日不得晚於結束日；查無資料時會報錯。
2. **數值轉換及補值**：v1 將 6 種污染物轉為數值並線性插值，不對目標 AQI 補值；v2 對 AQI 與 6 種污染物全部轉為浮點數並線性插值。v2 另會拒絕缺少必要欄位、重複時間或插值後仍有空值的資料。
3. **依時間切分**：前 80% 資料列作訓練集，後 20% 作測試集。`StandardScaler` 僅在訓練集的輸入特徵上 `fit`，測試集只 `transform`。v2 在縮放前把 AQI 複製到 `_target_aqi`，供原尺度標籤使用。
4. **建立逐時序列**：v1 要求歷史窗口及下一小時目標連續；v2 要求歷史窗口及全部未來目標連續。缺少的整點只會被統計，不會新增資料列；跨越時間缺口的序列會跳過。
5. **銜接測試歷史**：將訓練集最後 `window_size` 列接到測試集前端，讓測試集開頭也能使用前段歷史。測試標籤只取測試期間；v2 每筆樣本還必須有完整的未來 `forecast_horizon` 個目標。

目前線性插值發生在時間切分之前，可能使用缺口後方或切分點另一側的觀測值。v2 的訓練標籤及測試評估也可能包含插值後的 AQI，因此 pipeline 的測試指標與只使用原始真實值的回測／Gradio 評估口徑不同。

### 執行訓練

請在專案根目錄執行。以下兩個範例都使用木柵站 2025 年全年資料，包含 `2025-12-31` 全天。

v1：訓練下一小時 AQI 模型。

```powershell
python -m pipeline_v1.run_model `
  --station 木柵站 --start 2025-01-01 --end 2025-12-31 `
  --epochs 50 --window-size 24 --model aqi_lstm_model.pth --no-visualization
```

v2：訓練一次預測未來 8 小時 AQI 的模型。

```powershell
python -m pipeline_v2.run_model_v2 `
  --station 木柵站 --start 2025-01-01 --end 2025-12-31 `
  --epochs 50 --window-size 24 --forecast-horizon 8 `
  --model aqi_lstm_8h_model.pth
```

| 命令列參數 | v1 預設 | v2 預設 | 說明 |
|---|---|---|---|
| `--station` | `木柵站` | `木柵站` | 資料庫測站名稱 |
| `--start` | `2025-01-01` | `2025-01-01` | 開始日期，包含當日 |
| `--end` | `2025-12-31` | `2025-12-31` | 結束日期，包含當日全天 |
| `--model` | `aqi_lstm_model.pth` | `aqi_lstm_8h_model.pth` | checkpoint 輸出路徑，相對路徑以執行目錄為準 |
| `--epochs` | `50` | `50` | 訓練輪數 |
| `--window-size` | `24` | `24` | 歷史窗口小時數 |
| `--forecast-horizon` | 不提供 | `8` | v2 一次輸出的未來小時數 |
| `--no-visualization` | 未啟用 | 不提供 | v1 加上此旗標才會略過訓練前的 Plotly 圖表 |

v1 預設會開啟 AQI 趨勢圖與特徵散佈矩陣；v2 沒有內建視覺化步驟。v2 訓練可調整 `--forecast-horizon`，但根目錄的 `predict_8h.py` 僅接受 `forecast_horizon=8` 的 checkpoint。

兩個流程預設使用 Adam、learning rate `0.001`、MSE loss、batch size `32`；訓練 DataLoader 會打亂樣本，測試 DataLoader 不打亂。測試比例、batch size 與 learning rate 可由 Python 函式參數調整，未提供對應命令列旗標。

### 執行結果與 checkpoint

訓練入口依序執行資料清理、序列建立、訓練、測試評估、最新窗口預測及 checkpoint 儲存：

- v1 印出每輪 loss、測試集 MAE／RMSE，以及測試集最後窗口之後一小時的預測。
- v2 印出每輪 loss、各 horizon 的 MAE／RMSE 與「整體」指標，再列出測試集最後連續窗口之後的多步預測表。表格包含日期、預測 AQI 及可對應的實際 AQI；超出載入期間的實際值會為空。
- v2 的「整體 MAE／RMSE」分別為各 horizon 指標的算術平均；其中整體 RMSE 並非把所有 horizon 誤差攤平後重新開根號。
- 兩個訓練入口只讀取觀測資料、列印結果及儲存 checkpoint；要將預測寫入資料庫，請使用下節的 `predict_1h.py` 或 `predict_8h.py`。

checkpoint 包含 `model_state_dict`、`model_config`、`features`、`window_size`、`scaler_mean` 與 `scaler_scale`；v2 另保存 `forecast_horizon`，且 `model_config` 也包含該設定。推論時應沿用 checkpoint 的特徵順序、窗口與 scaler。只應載入可信任的本機 checkpoint。

## 歷史回測與未來預測

### v1

```powershell
python predict_1h.py --station 木柵站 --start 2026-01-01 --end 2026-08-01
```

v1 沒有 `--dry-run`，執行成功會寫入 `public.air_quality_predictions`。

### v2

```powershell
python predict_8h.py --station 木柵站 --start 2026-01-01 --end 2026-08-01
python predict_8h.py --station 木柵站 --start 2026-01-01 --end 2026-08-01 --dry-run
```

一般回測會為每個目標時間儲存可取得的 h1～h8。`model_version` 是 checkpoint 檔案的 SHA-256，避免不同模型結果互相覆蓋。

若要從最新有效觀測產生未來 8 小時：

```powershell
python predict_8h.py `
  --station 木柵站 `
  --start "2026-08-01 00:00" `
  --latest-origin "2026-10-11 13:00" `
  --end "2026-10-11 22:00"
```

使用 `--latest-origin` 時，`--end` 必須等於 `latest-origin + 9 小時`，才能讓半開區間包含最後一個 h8 目標。

### 使用現成檔案或記憶體資料

```powershell
python predict_1h.py --data downloaded.csv
python predict_8h.py --data batch1.ods batch2.xlsx --dry-run
```

支援 CSV、ODS、XLSX、XLS。若呼叫 `main(data=...)`，可傳入 DataFrame 或爬蟲回傳列；`data` 與 `--data` 不可同時指定。明確傳入空資料時不會偷偷回查資料庫。

推論階段只以前向補值準備模型特徵；原始真實 AQI 的缺值仍保留，不會把補值內容當成評估真值。v2 回測還會拒絕在觀測首尾不足時向外延伸資料。

## PGVector RAG 知識庫

先確保 PostgreSQL 可使用 `vector` extension，並讓 Ollama 的 embedding 模型保持運作：

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

建立或更新知識庫：

```powershell
python build_pgvector.py
```

目前知識庫固定使用：

- collection：`aqi_knowledge_base`
- 測站：`木柵站`
- 年份：2026
- 預測來源：`air_quality_predictions_8h`
- 文件：每月摘要與每日摘要
- embedding：預設 `nomic-embed-text`

每日摘要包括實際最高／最低 AQI、所有並列時間、有效觀測筆數、資料截止時間、平均實際／預測 AQI、MAE、RMSE、Bias、最大絕對誤差及走勢相反比例。最高／最低值使用木柵站當日所有有效觀測；平均及誤差只使用同時具有實際與預測值的配對樣本。

每月摘要的平均與誤差同樣只使用有效預測配對，因此不一定等於原始觀測表全部非空 AQI 的平均。每個目標時間會優先選最新 `created_at` 的 v2 預測；若時間相同，再依較小 `horizon_hours` 及 `model_version` 排序。

跨月製圖可輸入「將 2026/1~9月的平均 AQI 轉成長條圖」。知識庫工具會完整讀取範圍內的月摘要，依月份排序並檢查缺月、缺平均值，不受相似搜尋的 `top_k` 限制；缺資料時不會補零。智能助理的「查看圖表」按鈕固定顯示，成功製圖後會自動展開圖表區；模型回傳空白時會顯示提示。

文件 ID 使用 `source + station + date/month` 產生固定 UUID，所以相同規則下重跑會更新或新增文件。然而，如果文件 ID／metadata 規則曾改變，或來源資料已刪除，一般 `add_documents()` 不會自動清理失去對應關係的舊文件。

### 完整刪除此 collection 後重建

以下 SQL 只刪除 `aqi_knowledge_base`，不影響其他 PGVector collection：

```sql
BEGIN;

DELETE FROM public.langchain_pg_embedding AS e
USING public.langchain_pg_collection AS c
WHERE e.collection_id = c.uuid
  AND c.name = 'aqi_knowledge_base';

DELETE FROM public.langchain_pg_collection
WHERE name = 'aqi_knowledge_base';

COMMIT;
```

刪除後重新執行：

```powershell
python build_pgvector.py
```

不要直接 `TRUNCATE` 兩張 LangChain 資料表，除非確定要刪除資料庫中的所有 PGVector collections。

## 智能助理與工具

智能助理位於 Gradio 的「智能助理」頁面。每個使用者 session 各自保存對話歷史；清除對話或重啟服務後不保留。Agent 最多接受 4,000 字問題，回答以繁體中文呈現。

### AQIKnowledgeBase

- 查詢 2026 年木柵每日／每月預測評估摘要。
- 一般問題使用 PGVector 相似度檢索，固定過濾 `station=木柵站`。
- 指定「某年月每日平均 AQI」時，不使用 `top_k` 拼湊序列，而是直接從 PGVector 資料表完整讀取該月每日摘要、排序並檢查缺日。
- 會將實際使用的摘要顯示在「查看參考摘要」。
- 不適合查即時 AQI、未來天氣或逐筆原始資料。

### SqlTool

- 語言模型只負責產生結構化 `QueryPlan`；程式依 ORM 白名單組成 SQLAlchemy `SELECT`。
- 僅允許三張 AQI 資料表、單表查詢、白名單欄位與 `count/avg/min/max/sum`。
- 強制木柵站條件，每輪最多 3 個查詢，每個最多 500 列。
- PostgreSQL transaction 設為唯讀，statement timeout 10 秒、lock timeout 2 秒，結束一律 rollback。
- 不支援寫入、任意 SQL、JOIN、日期截斷、MAE／RMSE 或每組最新列。
- 建議用 `SQL_AGENT_DATABASE_URL` 配置只具有三張資料表 `SELECT` 權限的帳號。

### MuzhaWeatherForecast

- 固定座標為臺北市文山區木柵。
- 直接呼叫 Open-Meteo Forecast API，不使用 LLM 產生天氣數值。
- 提供目前氣溫、體感溫度、濕度、降雨及風況，以及今天起 1～7 天逐日預報。
- 不提供其他地點、歷史天氣或 AQI 推算。

### MatplotlibTool

- 接收 Agent 已取得的數值與繪圖需求，不自行讀資料庫或上網。
- 模型產生完整 Python 程式後，以獨立子程序執行；失敗時會把錯誤交回模型修正一次。
- 成功後在 `charts/` 留下 `chart_<id>.py` 與 `chart_<id>.png`，並於 Gradio Gallery 顯示圖片。
- 目前安全限制主要由 prompt 約束，並不是作業系統層級的完整沙箱；只應搭配可信任的模型服務與執行環境。

智能助理不能更新資料庫、重新訓練模型或產生新的 AQI 預測。資料庫中的「最新」只表示目前已儲存的最新時間，不等於即時資料。

## 評估口徑

令 `e = 預測 AQI - 實際 AQI`：

| 指標 | 定義 | 解讀 |
|---|---|---|
| MAE | `mean(abs(e))` | 越小越好 |
| RMSE | `sqrt(mean(e²))` | 對較大誤差更敏感，越小越好 |
| Bias | `mean(e)` | 正值代表高估，負值代表低估 |

圖表比較兩個模型時，只使用兩邊共同存在且預測、實際值皆有限的時間點。RAG 每日／每月摘要則使用其 SQL 查詢所選出的最新 v2 預測配對；兩者統計口徑不可直接假設完全相同。

## 常見問題

### `Failed to connect to Ollama`

```powershell
ollama serve
ollama list
```

確認 `LLM_BASE_URL`、服務連接埠與 `nomic-embed-text` 是否存在。建立向量與每次一般 RAG 查詢都會使用 embedding，因此都需要服務可連線。

### Agent 顯示授權失敗

目前 Agent 使用 Ollama Cloud 模型。確認 `.env` 中的 `OLLAMA_API_KEY` 有效，且目前帳號可使用 `agent/agent.py` 指定的模型。

### PostgreSQL 連線或建表失敗

確認：

- `DATABASE_URL` 使用 PostgreSQL 且包含主機、帳號與資料庫名稱。
- 資料庫已存在。
- 寫入帳號具有 `public` schema 建表及寫入權限。
- RAG 建置帳號能建立或使用 `vector` extension。
- `SQL_AGENT_DATABASE_URL` 若有設定，該帳號至少具有三張 AQI 資料表的 `SELECT` 權限。

### `langchain-community` DeprecationWarning

`build_pgvector.py` 目前仍使用 `langchain_community.utilities.SQLDatabase`，警告本身不會阻止執行。日後可改用獨立 integration package 或直接以 SQLAlchemy 執行摘要 SQL。

### 圖表中文字型缺失

Matplotlib 工具依序嘗試 `Noto Sans CJK TC`、`Microsoft JhengHei`、`PingFang TC`、`Arial Unicode MS` 與 `DejaVu Sans`。若 PNG 中文仍顯示方框，請在執行主機安裝其中一種中文字型。

### 更新流程中途失敗

爬蟲每個日期批次各自提交，而預測寫入採 upsert，因此可以再次按「更新至最新」。介面會保留原圖並顯示失敗階段；詳細 traceback 只寫入終端機，避免把連線資訊送到瀏覽器。

## 目前限制

- 智能助理、天氣與 RAG 固定服務木柵；不支援跨測站比較。
- RAG 摘要 SQL 固定為 2026 年，需修改程式才能建立其他年份。
- RAG 摘要來自 v2 預測，不代表固定單一 `horizon_hours` 或固定模型版本的評估。
- 氣象工具依賴 Open-Meteo 網路服務；Agent 回答與製圖依賴 Ollama Cloud。
- Matplotlib 工具會執行模型產生的 Python 程式，不適合在不受信任的多租戶主機直接開放。
- 專案尚未提供套件鎖定檔；重現環境時應自行固定已驗證的版本。
