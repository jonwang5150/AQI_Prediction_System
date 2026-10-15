"""AQI 唯讀 SQL agent；模型產生查詢規格，程式負責組成 SELECT。"""
from __future__ import annotations

import json
import os
from textwrap import dedent
from typing import Any, Literal
from chat_prompt import build_standard_chat_prompt_template
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import SQLAlchemyError

from config import settings
from db_models import AirQualityHourly, AirQualityPrediction, AirQualityPrediction8h

TABLES = {model.__tablename__: model.__table__ for model in (
    AirQualityHourly, AirQualityPrediction, AirQualityPrediction8h,
)}
TableName = Literal['air_quality_hourly', 'air_quality_predictions', 'air_quality_predictions_8h']


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Filter(StrictModel):
    column: str
    operator: Literal['eq', 'ne', 'gt', 'ge', 'lt', 'le', 'is_null', 'not_null']
    value: str | int | float | bool | None = None


class Metric(StrictModel):
    function: Literal['count', 'avg', 'min', 'max', 'sum']
    column: str


class QuerySpec(StrictModel):
    table: TableName
    columns: list[str] = Field(default_factory=list, max_length=20)
    filters: list[Filter] = Field(default_factory=list, max_length=20)
    metrics: list[Metric] = Field(default_factory=list, max_length=10)
    group_by: list[str] = Field(default_factory=list, max_length=10)
    order_by: str | None = None
    descending: bool = False
    limit: int = Field(default=100, ge=1, le=500)


class QueryPlan(StrictModel):
    refusal: str = ''
    queries: list[QuerySpec] = Field(default_factory=list, max_length=3)


def build_select(spec: QuerySpec):
    """識別字只能來自 ORM 白名單；條件值綁定為參數，不拼接 SQL。"""
    spec = QuerySpec.model_validate(spec.model_dump())
    table = TABLES[spec.table]

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
            expression = getattr(field, operator)(condition.value)
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
只能讀取以下 public 資料表。禁止修改、刪除、新增、建表或存取其他資料。
若要求寫入、任意 SQL、其他資料表或不支援的運算，填 refusal 並讓 queries 為空。
只使用提供的欄位；filters 之間為 AND。日期區間使用 ge 起點、lt 終點。
日期採台灣當地時間。統計必須使用 metrics，不得以截斷明細宣稱完整統計。
metrics 輸出名稱為 function_column，例如 avg_aqi；count(*) 為 count_all。
每次最多三個單表查詢，支援篩選、排序、分組與 count/avg/min/max/sum。
不支援跨表 JOIN、任意 SQL 運算式、MAE/RMSE 或每組最新列；這些需求請明確拒絕。
air_quality_hourly 是實際觀測；air_quality_predictions 是單步預測。
air_quality_predictions_8h 包含不同 horizon_hours 和 model_version；
不可把不同提前小時或模型版本當成同一筆預測，條件不清楚時要求使用者指定。
資料表與欄位：
''' + '\n'.join(f'public.{name}: ' + ', '.join(
    f'{c.name} ({c.type})' for c in table.c) for name, table in TABLES.items())


class SQLInput(StrictModel):
    query: str = Field(min_length=1, max_length=4000, description='要查詢三張 AQI 資料表的問題')


class SQLTool(BaseTool):
    name: str = "SqlTool"
    description: str = dedent("""\
一個專門設計用於唯讀查詢 AQI（空氣品質）資料庫的工具。

當出現以下情況時應使用此工具：
- 使用者詢問 AQI 實際觀測數據
- 使用者詢問單步預測或 8 小時預測結果
- 需要對上述資料做篩選、排序、分組或基本統計

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

    @classmethod
    def create(cls, llm: Runnable, *, engine=None):
        input_ = {"system": {"template": SQL_TOOL_SYSTEM_PROMPT},
                  "human": {"template": "{user_query}",
                            "input_variable": ["user_query"]}}

        # build_standard_chat_prompt_template 沿用專案中既有的函式（需自行 import）
        chat_prompt_template = build_standard_chat_prompt_template(input_)

        pipeline = chat_prompt_template | llm.with_structured_output(QueryPlan)

        if engine is None:
            # 可指定只授予三張表 SELECT 的帳號；此模組不修改資料庫權限。
            engine = create_engine(os.getenv('SQL_AGENT_DATABASE_URL') or settings.database_url,
                                   pool_pre_ping=True)

        return cls(runnable=pipeline, engine=engine)

    def _run(self, query: str):
        if not query.strip():
            return json.dumps({'error': '請輸入查詢問題'}, ensure_ascii=False)
        try:
            raw = self.runnable.invoke({"user_query": query})
            plan = QueryPlan.model_validate(raw.model_dump() if isinstance(raw, QueryPlan) else raw)
            result = execute_plan(self.engine, plan)
        except (ValueError, SQLAlchemyError):
            # 不將可能含帳密、SQL 參數的底層例外傳給模型或使用者。
            result = {'error': '查詢規格不允許或資料庫查詢失敗，請確認條件與唯讀連線設定。'}
        return json.dumps(result, ensure_ascii=False, default=str)

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")