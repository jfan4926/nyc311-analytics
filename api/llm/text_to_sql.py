import duckdb
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "data/processed/nyc311.duckdb"

# 数据库schema描述，告诉LLM有哪些表和字段
SCHEMA = """
You have access to a DuckDB database with the following tables:

1. mart_complaint_trends
   - week (TIMESTAMP): week start date
   - borough (VARCHAR): NYC borough name
   - complaint_type (VARCHAR): type of complaint
   - total_complaints (INT): number of complaints
   - closed_complaints (INT): number of resolved complaints
   - avg_resolution_hours (FLOAT): average hours to resolve
   - median_resolution_hours (FLOAT): median hours to resolve
   - closure_rate_pct (FLOAT): percentage of complaints closed

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


def ask(question: str) -> dict:
    """Takes natural language question, returns SQL + results."""
    llm = ChatGroq(model="llama-3.1-8b-instant")
    chain = PROMPT | llm

    # 生成SQL
    sql = chain.invoke({
        "schema": SCHEMA,
        "question": question
    }).content.strip()

    # 执行SQL
    results = query_database(sql)

    return {
        "question": question,
        "sql": sql,
        "results": results
    }


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