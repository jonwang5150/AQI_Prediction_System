"""建立 AQI 預測評估的向量知識庫。"""

from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

from config import settings


MONTHLY_QUERY = """
WITH ranked_predictions AS (
    SELECT
        station,
        target_at,
        predicted_aqi,
        ROW_NUMBER() OVER (
            PARTITION BY station, target_at
            ORDER BY created_at DESC, horizon_hours ASC, model_version ASC
        ) AS prediction_rank
    FROM air_quality_predictions_8h
    WHERE target_at >= DATE '2026-01-01'
),
paired AS (
    SELECT
        a.station,
        a.observed_at,
        a.aqi AS actual_aqi,
        p.predicted_aqi
    FROM air_quality_hourly a
    JOIN ranked_predictions p
        ON a.station = p.station
       AND a.observed_at = p.target_at
       AND p.prediction_rank = 1
    WHERE a.aqi IS NOT NULL
      AND p.predicted_aqi IS NOT NULL
      AND a.observed_at >= DATE '2026-01-01'
      AND a.observed_at < DATE '2027-01-01'
)
SELECT
    DATE_TRUNC('month', observed_at) AS month,
    ROUND(AVG(actual_aqi)::numeric, 2) AS avg_actual,
    ROUND(AVG(predicted_aqi)::numeric, 2) AS avg_predicted,
    ROUND(AVG(ABS(predicted_aqi - actual_aqi))::numeric, 2) AS mae,
    ROUND(SQRT(AVG(POWER(predicted_aqi - actual_aqi, 2)))::numeric, 2) AS rmse,
    ROUND(AVG(predicted_aqi - actual_aqi)::numeric, 2) AS bias,
    COUNT(*) AS sample_count
FROM paired
GROUP BY DATE_TRUNC('month', observed_at)
ORDER BY month
"""


DAILY_QUERY = """
WITH ranked_predictions AS (
    SELECT
        station,
        target_at,
        predicted_aqi,
        ROW_NUMBER() OVER (
            PARTITION BY station, target_at
            ORDER BY created_at DESC, horizon_hours ASC, model_version ASC
        ) AS prediction_rank
    FROM air_quality_predictions_8h
    WHERE target_at >= DATE '2026-01-01'
),
paired AS (
    SELECT
        a.station,
        a.observed_at,
        a.aqi::double precision AS actual_aqi,
        p.predicted_aqi::double precision AS predicted_aqi
    FROM air_quality_hourly a
    JOIN ranked_predictions p
        ON a.station = p.station
       AND a.observed_at = p.target_at
       AND p.prediction_rank = 1
    WHERE a.aqi IS NOT NULL
      AND p.predicted_aqi IS NOT NULL
      AND a.observed_at >= DATE '2026-01-01'
      AND a.observed_at < DATE '2027-01-01'
),
daily_changes AS (
    SELECT
        *,
        actual_aqi - LAG(actual_aqi) OVER (
            PARTITION BY station, DATE(observed_at)
            ORDER BY observed_at
        ) AS actual_change,
        predicted_aqi - LAG(predicted_aqi) OVER (
            PARTITION BY station, DATE(observed_at)
            ORDER BY observed_at
        ) AS predicted_change,
        LAG(observed_at) OVER (
            PARTITION BY station, DATE(observed_at)
            ORDER BY observed_at
        ) AS previous_time
    FROM paired
)
SELECT
    DATE(observed_at) AS date,
    ROUND(AVG(actual_aqi)::numeric, 2) AS avg_actual,
    ROUND(AVG(predicted_aqi)::numeric, 2) AS avg_predicted,
    ROUND(AVG(ABS(predicted_aqi - actual_aqi))::numeric, 2) AS mae,
    ROUND(SQRT(AVG(POWER(predicted_aqi - actual_aqi, 2)))::numeric, 2) AS rmse,
    ROUND(AVG(predicted_aqi - actual_aqi)::numeric, 2) AS bias,
    ROUND(MAX(ABS(predicted_aqi - actual_aqi))::numeric, 2) AS max_absolute_error,
    COUNT(*) AS sample_count,
    COUNT(*) FILTER (
        WHERE previous_time = observed_at - INTERVAL '1 hour'
          AND actual_change <> 0
          AND predicted_change <> 0
    ) AS valid_direction_pairs,
    COUNT(*) FILTER (
        WHERE previous_time = observed_at - INTERVAL '1 hour'
          AND actual_change * predicted_change < 0
    ) AS opposite_direction_count
FROM daily_changes
GROUP BY DATE(observed_at)
ORDER BY date
"""


def _build_monthly_documents(rows) -> list[Document]:
    documents = []
    for row in rows:
        content = f"""
月份：{row['month']}
平均實際 AQI：{row['avg_actual']}
平均預測 AQI：{row['avg_predicted']}
平均絕對誤差（MAE）：{row['mae']}
均方根誤差（RMSE）：{row['rmse']}
平均偏差（預測值減實際值）：{row['bias']}
配對樣本數：{row['sample_count']}
""".strip()
        documents.append(Document(
            page_content=content,
            metadata={
                "source": "aqi_prediction_monthly_evaluation",
                "prediction_table": "air_quality_predictions_8h",
                "prediction_selection": "latest_created_at",
                "month": str(row["month"]),
            },
        ))
    return documents


def _build_daily_documents(rows) -> list[Document]:
    documents = []
    for row in rows:
        valid_pairs = row["valid_direction_pairs"]
        opposite_count = row["opposite_direction_count"]
        opposite_ratio = (
            round(opposite_count / valid_pairs * 100, 2)
            if valid_pairs > 0 else None
        )
        ratio_text = opposite_ratio if opposite_ratio is not None else "無法計算"
        content = f"""
日期：{row['date']}
平均實際 AQI：{row['avg_actual']}
平均預測 AQI：{row['avg_predicted']}
平均絕對誤差（MAE）：{row['mae']}
均方根誤差（RMSE）：{row['rmse']}
平均偏差（Bias）：{row['bias']}
最大絕對誤差：{row['max_absolute_error']}
有效配對樣本數：{row['sample_count']}
有效方向比較次數：{valid_pairs}
走勢相反次數：{opposite_count}
走勢相反比例：{ratio_text}%
""".strip()
        documents.append(Document(
            page_content=content,
            metadata={
                "source": "aqi_prediction_daily_evaluation",
                "prediction_table": "air_quality_predictions_8h",
                "prediction_selection": "latest_created_at",
                "type": "daily_summary",
                "date": str(row["date"]),
                "year": row["date"].year,
                "month": row["date"].month,
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "bias": float(row["bias"]),
                "sample_count": int(row["sample_count"]),
                "opposite_ratio": opposite_ratio,
            },
        ))
    return documents


def _document_id(document: Document) -> str:
    """為同一份統計文件產生固定 ID，重建時更新而不重複新增。"""
    metadata = document.metadata
    period = metadata.get("date", metadata.get("month", ""))
    key = f"{metadata['source']}:{period}"
    return str(uuid5(NAMESPACE_URL, key))


def build_knowledge_base(
    database_url: str | None = None,
    *,
    embedding_model: str | None = None,
    ollama_base_url: str | None = None,
    collection_name: str = "aqi_knowledge_base",
) -> int:
    """查詢評估資料、建立文件並寫入向量庫，回傳寫入文件數。"""
    connection = database_url or settings.database_url
    database = SQLDatabase.from_uri(connection)

    monthly_rows = database.run(
        MONTHLY_QUERY, fetch="cursor"
    ).mappings().all()
    daily_rows = database.run(
        DAILY_QUERY, fetch="cursor"
    ).mappings().all()
    documents = (
        _build_monthly_documents(monthly_rows)
        + _build_daily_documents(daily_rows)
    )

    if not documents:
        return 0

    embeddings = OllamaEmbeddings(
        model=embedding_model or settings.ollama_embedding_model,
        base_url=ollama_base_url or settings.ollama_base_url,
    )
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )
    vector_store.add_documents(
        documents,
        ids=[_document_id(document) for document in documents],
    )
    return len(documents)


def main() -> int:
    document_count = build_knowledge_base()
    print(f"向量知識庫建立完成，共寫入 {document_count} 份 Document")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
