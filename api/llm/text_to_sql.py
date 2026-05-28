import os
import sys
import duckdb
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from api.llm.rag import retrieve_context

DB_PATH = "data/processed/nyc311.duckdb"

# 数据库schema描述，告诉LLM有哪些表和字段
SCHEMA = """
You have access to a DuckDB database with the following tables:

1. mart_complaint_trends  
   - week (TIMESTAMP): week start date  ← this table has ONE ROW PER WEEK PER BOROUGH PER COMPLAINT_TYPE
   - borough (VARCHAR): NYC borough name
   - complaint_type (VARCHAR): type of complaint
   - total_complaints (INT): complaint count for that week
   - closed_complaints (INT): resolved complaints for that week
   - avg_resolution_hours (FLOAT): average hours to resolve
   - median_resolution_hours (FLOAT): median hours to resolve
   - closure_rate_pct (FLOAT): percentage of complaints closed
   
   IMPORTANT: To get totals or averages across all time, always use aggregation:
   Example - top complaint types by volume:
   SELECT complaint_type, SUM(total_complaints) AS total FROM mart_complaint_trends GROUP BY complaint_type ORDER BY total DESC LIMIT 5
   Example - closure rate by borough:
   SELECT borough, ROUND(AVG(closure_rate_pct),1) AS avg_closure_rate FROM mart_complaint_trends GROUP BY borough ORDER BY avg_closure_rate DESC

2. mart_agency_performance
   - agency (VARCHAR): agency code
   - agency_name (VARCHAR): full agency name
   - complaint_type (VARCHAR): type of complaint
   - total_complaints (INT)
   - avg_resolution_hours (FLOAT)
   - median_resolution_hours (FLOAT)
   - p90_resolution_hours (FLOAT): 90th percentile resolution hours
   - pct_over_1_week (FLOAT): percentage taking more than 1 week

3. mart_ml_features
   - unique_key (VARCHAR): primary key
   - created_at (TIMESTAMP)
   - resolution_hours (FLOAT): hours to resolve, null if open
   - agency (VARCHAR)
   - complaint_type (VARCHAR)
   - borough (VARCHAR)
   - channel_type (VARCHAR): how complaint was submitted
   - created_hour (INT): hour of day created
   - created_dow (INT): day of week (0=Sunday)
   - created_month (INT)
   - is_weekend (BOOL)
   - hist_avg_resolution_hours (FLOAT)
   - is_overdue (INT): 1 if took more than 1 week, 0 otherwise
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a SQL expert. Given a question, write a DuckDB SQL query to answer it.

{schema}

Rules:
- Return ONLY the SQL query, no explanation, no markdown, no backticks
- Use only the tables listed above
- Always use LIMIT not TOP for limiting rows (this is DuckDB not SQL Server)
- Always add LIMIT 10 for non-aggregation queries
- For aggregation queries (GROUP BY), no LIMIT needed unless specified
- For borough names use uppercase (BROOKLYN, QUEENS, MANHATTAN, BRONX, STATEN ISLAND)
- Always SELECT meaningful columns, not just one column unless asked
"""),
    ("human", "{question}")
])


def query_database(sql: str) -> str:
    """Execute SQL and return results as string."""
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        df = con.execute(sql).df()
        con.close()
        if df.empty:
            return "No results found."
        return df.to_string(index=False)
    except Exception as e:
        return f"SQL Error: {e}"



#RAG Version
def ask(question: str) -> dict:
    """Takes natural language question, returns SQL + results."""
    llm = ChatGroq(model="llama-3.1-8b-instant")

    # RAG: 检索相关schema和示例
    context = retrieve_context(question)

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a SQL expert for a NYC 311 complaints database.

Use the following relevant schema and examples to write accurate DuckDB SQL:

{context}

Rules:
- Return ONLY the SQL query, no explanation, no markdown, no backticks
- Use only the tables shown above
- Always use LIMIT not TOP
- For mart_complaint_trends: it has one row per week per borough per complaint_type, so always use SUM() or AVG() with GROUP BY for totals
- For borough names use uppercase (BROOKLYN, QUEENS, MANHATTAN, BRONX, STATEN ISLAND)
- For aggregation queries no LIMIT needed unless specified
"""),
        ("human", "{question}")
    ])

    chain = prompt | llm
    sql   = chain.invoke({"question": question}).content.strip()
    results = query_database(sql)

    return {
        "question": question,
        "sql":      sql,
        "results":  results
    }
# def ask(question: str) -> dict:
#     """Takes natural language question, returns SQL + results."""
#     llm = ChatGroq(model="llama-3.1-8b-instant")
#     chain = PROMPT | llm

#     # 生成SQL
#     sql = chain.invoke({
#         "schema": SCHEMA,
#         "question": question
#     }).content.strip()

#     # 执行SQL
#     results = query_database(sql)

#     return {
#         "question": question,
#         "sql": sql,
#         "results": results
#     }


if __name__ == "__main__":
    # 测试几个问题
    questions = [
        "Which borough has the most complaints?",
        "What are the top 5 complaint types by average resolution time?",
        "Which agency has the worst performance (highest pct_over_1_week)?"
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = ask(q)
        print(f"SQL: {result['sql']}")
        print(f"Results:\n{result['results']}")