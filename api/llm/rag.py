import chromadb
from chromadb.utils import embedding_functions
import duckdb
import os

DB_PATH    = "data/processed/nyc311.duckdb"
CHROMA_DIR = "api/llm/chroma_store"

# 用轻量本地embedding模型，不需要API
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name="nyc311_knowledge",
        embedding_function=ef
    )

def build_knowledge_base():
    """
    一次性建立向量库，存入：
    1. 每张mart表的schema描述
    2. 常用业务问题的示例SQL
    """
    col = get_collection()

    # ── Schema知识 ──────────────────────────────────────
    schema_docs = [
        {
            "id": "schema_trends",
            "text": """Table mart_complaint_trends stores weekly aggregated complaint data.
Each row = one week + one borough + one complaint_type combination.
Columns: week (TIMESTAMP), borough, complaint_type, total_complaints (INT),
closed_complaints (INT), avg_resolution_hours (FLOAT),
median_resolution_hours (FLOAT), closure_rate_pct (FLOAT).
To get all-time totals use SUM() with GROUP BY complaint_type or borough."""
        },
        {
            "id": "schema_agency",
            "text": """Table mart_agency_performance stores agency SLA metrics per complaint type.
Each row = one agency + one complaint_type combination.
Columns: agency (VARCHAR), agency_name (VARCHAR), complaint_type,
total_complaints (INT), avg_resolution_hours (FLOAT),
median_resolution_hours (FLOAT), p90_resolution_hours (FLOAT),
pct_over_1_week (FLOAT - percentage of complaints taking more than 7 days).
Only includes combinations with at least 50 complaints."""
        },
        {
            "id": "schema_features",
            "text": """Table mart_ml_features is the ML feature table, one row per complaint.
Columns: unique_key, created_at (TIMESTAMP), resolution_hours (FLOAT, null if open),
agency, complaint_type, borough, channel_type,
created_hour (0-23), created_dow (0=Sunday), created_month (1-12),
is_weekend (BOOL), hist_avg_resolution_hours (FLOAT),
is_overdue (INT, 1 if resolution_hours > 168 else 0)."""
        },
    ]

    # ── 示例query知识 ────────────────────────────────────
    example_docs = [
        {
            "id": "ex_top_complaints",
            "text": """Question: top complaint types by volume
SQL: SELECT complaint_type, SUM(total_complaints) AS total
FROM mart_complaint_trends
GROUP BY complaint_type
ORDER BY total DESC LIMIT 5"""
        },
        {
            "id": "ex_borough_closure",
            "text": """Question: borough closure rate comparison
SQL: SELECT borough, ROUND(AVG(closure_rate_pct), 1) AS avg_closure_rate
FROM mart_complaint_trends
WHERE borough IS NOT NULL
GROUP BY borough
ORDER BY avg_closure_rate DESC"""
        },
        {
            "id": "ex_worst_agency",
            "text": """Question: worst agency resolution time
SQL: SELECT agency_name, ROUND(AVG(avg_resolution_hours), 1) AS avg_hours
FROM mart_agency_performance
GROUP BY agency_name
ORDER BY avg_hours DESC LIMIT 5"""
        },
        {
            "id": "ex_overdue_rate",
            "text": """Question: which complaint types are most likely to be overdue or delayed
SQL: SELECT complaint_type,
     ROUND(AVG(is_overdue) * 100, 1) AS overdue_pct,
     COUNT(*) AS total
FROM mart_ml_features
WHERE is_overdue IS NOT NULL
GROUP BY complaint_type
HAVING COUNT(*) >= 100
ORDER BY overdue_pct DESC LIMIT 10"""
        },
        {
            "id": "ex_weekly_trend",
            "text": """Question: weekly trend or volume over time
SQL: SELECT week, SUM(total_complaints) AS total_complaints
FROM mart_complaint_trends
GROUP BY week
ORDER BY week"""
        },
    ]

    all_docs = schema_docs + example_docs
    col.upsert(
        ids=[d["id"] for d in all_docs],
        documents=[d["text"] for d in all_docs]
    )
    print(f"Knowledge base built: {len(all_docs)} documents")


def retrieve_context(question: str, n_results: int = 3) -> str:
    """
    Given a natural language question, retrieve the most relevant
    schema descriptions and example SQLs from the vector store.
    """
    col     = get_collection()
    results = col.query(query_texts=[question], n_results=n_results)
    docs    = results["documents"][0]
    return "\n\n---\n\n".join(docs)


if __name__ == "__main__":
    build_knowledge_base()
    # 测试检索
    q = "which borough has the most noise complaints?"
    print(f"\nQuery: {q}")
    print(retrieve_context(q))