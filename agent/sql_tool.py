"""AQI 唯讀 SQL agent；模型產生查詢規格，程式負責組成 SELECT。"""
from __future__ import annotations

import json
import asyncio
import os
from datetime import datetime, timedelta, timezone
from textwrap import dedent
from typing import Any, Literal
from .chat_prompt import build_standard_chat_prompt_template
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from langchain_core.exceptions import OutputParserException
from httpx import HTTPError
from ollama import ResponseError
from sqlalchemy import DateTime, create_engine, func, select
from sqlalchemy.exc import SQLAlchemyError

from config import settings
from db_models import AirQualityHourly, AirQualityPrediction, AirQualityPrediction8h

TABLES = {model.__tablename__: model.__table__ for model in (
    AirQualityHourly, AirQualityPrediction, AirQualityPrediction8h,
)}
TableName = Literal['air_quality_hourly', 'air_quality_predictions', 'air_quality_predictions_8h']


# 拒絕未定義的欄位，作為查詢規格的驗證基底。
class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


# 描述單一欄位的篩選運算與條件值。
class Filter(StrictModel):
    column: str
    operator: Literal['eq', 'ne', 'gt', 'ge', 'lt', 'le', 'is_null', 'not_null']
    value: str | int | float | bool | None = None


# 定義允許的聚合函式及其統計欄位。
class Metric(StrictModel):
    function: Literal['count', 'avg', 'min', 'max', 'sum']
    column: str


# 描述單表查詢的欄位、條件、統計、排序與筆數限制。
class QuerySpec(StrictModel):
    table: TableName
    columns: list[str] = Field(default_factory=list, max_length=20)
    filters: list[Filter] = Field(default_factory=list, max_length=20)
    metrics: list[Metric] = Field(default_factory=list, max_length=10)
    group_by: list[str] = Field(default_factory=list, max_length=10)
    order_by: str | None = None
    descending: bool = False
    limit: int = Field(default=100, ge=1, le=500)


# 封裝最多三個查詢規格，或記錄無法執行的拒絕原因。
class QueryPlan(StrictModel):
    refusal: str = ''
    queries: list[QuerySpec] = Field(default_factory=list, max_length=3)


def build_select(spec: QuerySpec):
    """識別字只能來自 ORM 白名單；條件值綁定為參數，不拼接 SQL。"""
    spec = QuerySpec.model_validate(spec.model_dump())
    table = TABLES[spec.table]

    # 取得白名單內的資料表欄位，拒絕未知欄位名稱。
    def column(name):
        if name not in table.c:
            raise ValueError(f'不允許的欄位：{name}')
        return table.c[name]

    groups = [column(name) for name in spec.group_by]
    if spec.metrics:
        if spec.columns:
            raise ValueError('統計查詢請以 group_by 指定分組欄位，不可同時使用 columns')
        selected = list(groups)
        for metric in spec.metrics:
            if metric.column == '*':
                if metric.function != 'count':
                    raise ValueError('只有 count 可以使用 *')
                expression = func.count()
            else:
                expression = getattr(func, metric.function)(column(metric.column))
            label = metric.function + '_' + metric.column.replace('*', 'all')
            selected.append(expression.label(label))
    else:
        if groups:
            raise ValueError('group_by 必須搭配 metrics')
        selected = [column(name) for name in spec.columns] if spec.columns else list(table.c)
    names = [item.key for item in selected]
    if len(names) != len(set(names)):
        raise ValueError('查詢輸出欄位不可重複')
    statement = select(*selected).select_from(table)
    for condition in spec.filters:
        field = column(condition.column)
        if condition.operator == 'is_null':
            expression = field.is_(None)
        elif condition.operator == 'not_null':
            expression = field.is_not(None)
        else:
            if condition.value is None:
                raise ValueError('空值請使用 is_null 或 not_null')
            operator = {'eq': '__eq__', 'ne': '__ne__', 'gt': '__gt__',
                        'ge': '__ge__', 'lt': '__lt__', 'le': '__le__'}[condition.operator]
            value = condition.value
            if field.name == 'station' and value in ('木柵', '木柵站', '木柵觀測站'):
                value = '木柵站'
            if isinstance(field.type, DateTime):
                if not isinstance(value, str):
                    raise ValueError('日期條件必須使用 ISO 日期或時間字串')
                value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                taipei = timezone(timedelta(hours=8))
                if field.type.timezone:
                    if value.tzinfo is None:
                        value = value.replace(tzinfo=taipei)
                elif value.tzinfo is not None:
                    value = value.astimezone(taipei).replace(tzinfo=None)
            expression = getattr(field, operator)(value)
        statement = statement.where(expression)
    if groups:
        statement = statement.group_by(*groups)
    if spec.order_by:
        outputs = dict(zip(names, selected))
        if spec.order_by not in outputs:
            raise ValueError('排序欄位必須出現在查詢輸出中')
        order = outputs[spec.order_by]
        statement = statement.order_by(order.desc() if spec.descending else order.asc())
    return statement.limit(spec.limit + 1)


def execute_plan(engine, plan: QueryPlan) -> dict:
    """全部驗證後才連線；PostgreSQL 唯讀交易、逾時限制、結束時 rollback。"""
    plan = QueryPlan.model_validate(plan.model_dump())
    if plan.refusal:
        return {'refusal': plan.refusal, 'results': []}
    if not plan.queries:
        raise ValueError('沒有可執行的查詢')
    statements = [build_select(spec) for spec in plan.queries]
    if engine.dialect.name != 'postgresql':
        raise ValueError('SQL agent 僅支援 PostgreSQL 唯讀交易')
    results = []
    with engine.connect().execution_options(isolation_level='READ COMMITTED') as connection:
        transaction = connection.begin()
        try:
            connection.exec_driver_sql('SET TRANSACTION READ ONLY')
            connection.exec_driver_sql('SET LOCAL search_path = pg_catalog')
            connection.exec_driver_sql("SET LOCAL statement_timeout = '10s'")
            connection.exec_driver_sql("SET LOCAL lock_timeout = '2s'")
            for spec, statement in zip(plan.queries, statements):
                rows = connection.execute(statement).mappings().fetchall()
                results.append({'table': spec.table, 'query': spec.model_dump(),
                                'rows': [dict(row) for row in rows[:spec.limit]],
                                'truncated': len(rows) > spec.limit})
        finally:
            transaction.rollback()
    return {'results': results}


SQL_TOOL_SYSTEM_PROMPT = '''你是 AQI 唯讀 SQL agent，將問題轉成 QueryPlan。
服務僅限木柵觀測站，未指定地點時預設木柵。資料庫測站名稱是「木柵站」，每個查詢必須包含 station eq 木柵站 的 filters 條件。
明確要求其他測站、全台或跨地區比較時，填 refusal 說明僅提供木柵資料，queries 保持空白。
必須呼叫 QueryPlan 工具提交查詢規格，不能只回覆文字或 SQL。
table 使用不含 public. 的表名。filters 每項只能使用 column、operator、value；
metrics 每項只能使用 function、column。禁止使用 field、op、as 等未定義鍵。
operator 使用 eq/ne/gt/ge/lt/le/is_null/not_null；函式名稱使用小寫。
只能讀取以下 public 資料表。禁止修改、刪除、新增、建表或存取其他資料。
若要求寫入、任意 SQL、其他資料表或不支援的運算，填 refusal 並讓 queries 為空。
只使用提供的欄位；filters 之間為 AND。日期區間使用 ge 起點、lt 終點。
日期採台灣當地時間。統計必須使用 metrics，不得以截斷明細宣稱完整統計。
單一日期的每日平均不需要日期截斷或 GROUP BY：篩選當日 00:00（含）至次日 00:00（不含），
使用 metrics 的 avg(aqi) 與 count(aqi)，columns 與 group_by 都保持空白。單一月份平均同理。
count(aqi) 是有效 AQI 樣本數；count(*) 是總列數。不能因「每日平均」字樣就拒絕單日聚合。
只有跨多日且要求按日期分組的查詢，才涉及目前不支援的日期截斷運算。
metrics 輸出名稱為 function_column，例如 avg_aqi；count(*) 為 count_all。
每次最多三個單表查詢，支援篩選、排序、分組與 count/avg/min/max/sum。
不支援跨表 JOIN、任意 SQL 運算式、MAE/RMSE 或每組最新列；這些需求請明確拒絕。
air_quality_hourly 是實際觀測；air_quality_predictions 是單步預測。
air_quality_predictions_8h 包含不同 horizon_hours 和 model_version；
不可把不同提前小時或模型版本當成同一筆預測，條件不清楚時要求使用者指定。
資料表與欄位：
''' + '\n'.join(f'public.{name}: ' + ', '.join(
    f'{c.name} ({c.type})' for c in table.c) for name, table in TABLES.items())


# 驗證自然語言查詢問題，限制空值與輸入長度。
class SQLInput(StrictModel):
    query: str = Field(min_length=1, max_length=4000, description='要查詢三張 AQI 資料表的問題')


# 將自然語言轉為受限查詢規格，並以唯讀交易取得資料。
class SQLTool(BaseTool):
    name: str = "SqlTool"
    description: str = dedent("""\
一個專門設計用於唯讀查詢木柵觀測站 AQI（空氣品質）資料庫的工具。僅服務木柵，查詢須限定 station 等於木柵站。
單純的日期／月份摘要問題（例如使用者指定月份的平均 AQI）優先使用 AQIKnowledgeBase。
本工具用於逐筆資料、自訂篩選或明確要求從原始觀測重新計算的統計，不應與已足夠的摘要重複查詢。

當出現以下情況時應使用此工具：
- 使用者詢問逐筆 AQI 實際觀測、特定測站或最新紀錄
- 使用者詢問單步預測或 8 小時預測結果
- 需要對上述原始資料做自訂篩選、排序、分組或明確要求完整原始觀測的基本統計

此工具負責：
- 將問題轉換為結構化查詢規格，並以唯讀方式查詢資料庫
- 支援單表篩選、排序、分組，以及 count / avg / min / max / sum 統計
- 回傳 JSON 格式的查詢結果

輸入內容應清楚說明：
- 要查詢的資料類型（實際觀測、單步預測或 8h 預測）
- 時間範圍、測站或其他篩選條件
- 需要的統計方式（若有）

此工具不支援寫入、跨表 JOIN、任意 SQL、MAE/RMSE 或每組最新列等運算，
也不負責對話管理或任務編排。
""")
    args_schema: type[BaseModel] = SQLInput

    runnable: Runnable
    engine: Any = Field(exclude=True)

    # 組合結構化查詢模型與資料庫引擎，建立 SQL 工具。
    @classmethod
    def create(cls, llm: Runnable, *, engine=None):
        example = QueryPlan(queries=[QuerySpec(
            table='air_quality_hourly',
            filters=[Filter(column='station', operator='eq', value='木柵站')],
            metrics=[Metric(function='avg', column='aqi')],
        )]).model_dump_json()
        input_ = {"system": {
                    "template": SQL_TOOL_SYSTEM_PROMPT + '\n完整 QueryPlan JSON Schema：\n{plan_schema}'
                                '\n無日期限制的格式範例；實際查詢必須依使用者指定的年月加入對應日期篩選：\n{plan_example}',
                    "partial_variables": {"plan_schema": json.dumps(QueryPlan.model_json_schema(), ensure_ascii=False),
                                          "plan_example": example}},
                  "human": {"template": "{user_query}",
                            "input_variables": ["user_query"]}}

        # build_standard_chat_prompt_template 沿用專案中既有的函式（需自行 import）
        chat_prompt_template = build_standard_chat_prompt_template(input_)

        # Cloud supports tool calling, but not Ollama's native JSON-schema format.
        pipeline = chat_prompt_template | llm.with_structured_output(QueryPlan, method="function_calling")

        if engine is None:
            # 可指定只授予三張表 SELECT 的帳號；此模組不修改資料庫權限。
            engine = create_engine(os.getenv('SQL_AGENT_DATABASE_URL') or settings.database_url,
                                   pool_pre_ping=True)

        return cls(runnable=pipeline, engine=engine)

    # 產生並驗證查詢計畫，執行後回傳結果或明確的錯誤資訊。
    def _run(self, query: str):
        if not query.strip():
            return json.dumps({'error': '請輸入查詢問題'}, ensure_ascii=False)
        stage = 'plan'
        try:
            # Repair malformed model output once, without weakening validation.
            for attempt in range(2):
                try:
                    prompt = query if attempt == 0 else (
                        query + '\n上次輸出未符合 QueryPlan。請依系統提供的完整 schema 與範例重建，'
                        '呼叫 QueryPlan；table 不加 public.，使用 column/operator/value 與 function/column。'
                    )
                    raw = self.runnable.invoke({"user_query": prompt})
                    plan = QueryPlan.model_validate(raw.model_dump() if isinstance(raw, QueryPlan) else raw)
                    break
                except (ValidationError, OutputParserException):
                    if attempt == 1:
                        raise
            stage = 'execute'
            result = execute_plan(self.engine, plan)
        except (HTTPError, ResponseError, ConnectionError):
            result = {'error': 'SQL 子模型服務或授權失敗，請停止重試並檢查模型連線設定。',
                      'error_code': 'model_service_error', 'retryable': False}
        except SQLAlchemyError:
            result = {'error': '資料庫連線或查詢執行失敗，請停止重試並檢查資料庫設定。',
                      'error_code': 'database_error', 'retryable': False}
        except ValueError:
            result = {'error': ('SQL 子模型未產生有效 QueryPlan，已修正重試一次仍失敗，請停止重試。'
                                if stage == 'plan' else '查詢不符合支援的欄位或運算規則。'),
                      'error_code': 'invalid_query_plan' if stage == 'plan' else 'unsupported_query',
                      'retryable': False}
        return json.dumps(result, ensure_ascii=False, default=str)

    # 以執行緒執行同步 SQL 查詢，支援非同步工具呼叫。
    async def _arun(self, query: str):
        return await asyncio.to_thread(self._run, query)
