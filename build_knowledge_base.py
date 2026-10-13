from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from config import settings

db = SQLDatabase.from_uri(settings.database_url)

print(db.get_usable_table_names())


month_result = db.run("""
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
""", fetch="cursor").mappings().all()
# print(result)



documents_month = []
for row in month_result:
    content = f"""
    月份：{row['month']}
    平均實際 AQI：{row['avg_actual']}
    平均預測 AQI：{row['avg_predicted']}
    平均絕對誤差（MAE）：{row['mae']}
    均方根誤差（RMSE）：{row['rmse']}
    平均偏差（預測值減實際值）：{row['bias']}
    配對樣本數：{row['sample_count']}
    """.strip()

    documents_month.append(
        Document(
            page_content=content,
            metadata={
                "source": "aqi_prediction_monthly_evaluation",
                "prediction_table": "air_quality_predictions_8h",
                "prediction_selection": "latest_created_at",
                "month": str(row["month"])
            }
        )
    )


# print(f"每月 Document 數量：{len(documents_month)}")
# for document in documents_month:
#     print(document)
#     print()




daily_result = db.run("""
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

    ROUND(AVG(actual_aqi)::numeric, 2)
        AS avg_actual,

    ROUND(AVG(predicted_aqi)::numeric, 2)
        AS avg_predicted,

    ROUND(
        AVG(ABS(predicted_aqi - actual_aqi))::numeric, 2
    ) AS mae,

    ROUND(
        SQRT(AVG(POWER(predicted_aqi - actual_aqi, 2)))::numeric, 2
    ) AS rmse,

    ROUND(
        AVG(predicted_aqi - actual_aqi)::numeric, 2
    ) AS bias,

    ROUND(
        MAX(ABS(predicted_aqi - actual_aqi))::numeric, 2
    ) AS max_absolute_error,

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
ORDER BY date;
""", fetch="cursor").mappings().all()


documents_daily = []

for row in daily_result:
    valid_pairs = row["valid_direction_pairs"]
    opposite_count = row["opposite_direction_count"]

    opposite_ratio = (
        round(opposite_count / valid_pairs * 100, 2)
        if valid_pairs > 0 else None
    )

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
走勢相反比例：{opposite_ratio if opposite_ratio is not None else '無法計算'}%
""".strip()

    documents_daily.append(
        Document(
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
                "opposite_ratio": opposite_ratio
            }
        )
    )

print(f"每日 Document 數量：{len(documents_daily)}")
# for document in documents_daily:
#     print(document.page_content)
#     print()



embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="aqi_knowledge_base",
    connection=settings.database_url,
    use_jsonb=True
)

# 2. 將 Document 向量化並儲存
documents = documents_month + documents_daily
if documents:
    vector_store.add_documents(documents)

print("向量知識庫建立完成")