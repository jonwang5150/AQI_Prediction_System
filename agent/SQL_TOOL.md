# AQI 唯讀 SQL agent

從專案根目錄使用已安裝 LangChain 依賴的 Python 環境：

```python
from agent.sql_tool import SQLTool

sql_tool = SQLTool.create()  # 沿用 config.py 的 Ollama 設定
result = sql_tool.invoke({"query": "各測站的實際 AQI 平均值是多少？"})
print(result)
```

也可以 `SQLTool.create(llm=your_model)` 指定支援 `with_structured_output()` 的模型，
或將回傳的工具加入 LangChain agent 的 tools 清單。

允許讀取 `public.air_quality_hourly`、`public.air_quality_predictions`、
`public.air_quality_predictions_8h`。支援單表篩選、排序、分組及
count/avg/min/max/sum；一次最多三個查詢，每個最多回傳 500 筆。
結果為 JSON，含查詢條件、資料列及 `truncated`，不是模型生成的敘述答案。
不支援 JOIN、任意 SQL、MAE/RMSE 或每組最新列。

安全限制由程式執行，不只依賴提示詞：結構化規格禁止額外欄位，
表名與欄位查白名單，值使用參數綁定，僅建立 SELECT。
每次使用 PostgreSQL READ ONLY 交易，限制單次 SQL 10 秒、鎖等待 2 秒，最後 rollback。

`SQL_AGENT_DATABASE_URL` 可指定獨立的資料庫唯讀帳號；未設定時沿用 `DATABASE_URL`。
如需在資料庫帳號層也強制只讀三張表，應使用僅具這三張表 SELECT 權限的非管理員帳號。
程式不會建立帳號或修改既有權限。

離線測試（不呼叫模型、不連資料庫）：

```powershell
.venv-rag/Scripts/python.exe -m unittest test_sql_tool -v
```

參考：[LangChain 結構化輸出](https://reference.langchain.com/python/langchain-core/language_models/chat_models/BaseChatModel/with_structured_output)、
[PostgreSQL 唯讀交易](https://www.postgresql.org/docs/15/sql-set-transaction.html)。
