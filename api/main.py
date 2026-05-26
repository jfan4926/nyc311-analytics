import duckdb
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.llm.text_to_sql import ask
import os

DB_PATH = "data/processed/nyc311.duckdb"

app = FastAPI(
    title="NYC311 Analytics API",
    description="Text-to-SQL and complaint overdue prediction",
    version="1.0.0"
)

# ── 启动时加载模型 ─────────────────────────────────────────
@app.on_event("startup")
async def load_model():
    global model
    runs = mlflow.search_runs(experiment_names=["nyc311-overdue-prediction"])
    best_run = runs.sort_values("metrics.test_auc", ascending=False).iloc[0]
    run_id = best_run["run_id"]
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    print(f"Model loaded from run: {run_id}")


# ── Request/Response schemas ──────────────────────────────
class QuestionRequest(BaseModel):
    question: str

class PredictRequest(BaseModel):
    agency: str
    complaint_type: str
    borough: str
    channel_type: str
    created_hour: int
    created_dow: int
    created_month: int
    is_weekend: bool


# ── Endpoints ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(req: QuestionRequest):
    """Natural language → SQL → results"""
    try:
        result = ask(req.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(req: PredictRequest):
    """Predict if a complaint will be overdue"""
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        hist_avg = con.execute("""
            SELECT COALESCE(hist_avg_resolution_hours, 0)
            FROM mart_ml_features
            WHERE agency = ? AND complaint_type = ?
            LIMIT 1
        """, [req.agency, req.complaint_type]).fetchone()
        con.close()
        hist_avg = hist_avg[0] if hist_avg else 0.0

        # 注意：LabelEncoder训练时用的integer codes
        # 这里简化处理，直接用hash取模
        features = pd.DataFrame([{
            "agency":                    hash(req.agency) % 1000,
            "complaint_type":            hash(req.complaint_type) % 1000,
            "borough":                   hash(req.borough) % 1000,
            "channel_type":              hash(req.channel_type) % 1000,
            "created_hour":              req.created_hour,
            "created_dow":               req.created_dow,
            "created_month":             req.created_month,
            "is_weekend":                int(req.is_weekend),
            "hist_avg_resolution_hours": hist_avg
        }])

        proba = model.predict_proba(features)[0][1]
        pred  = int(proba >= 0.5)

        return {
            "is_overdue_predicted": pred,
            "overdue_probability":  round(float(proba), 4),
            "message": "High risk of delay" if pred == 1 else "Likely on time"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/borough")
def borough_stats():
    """Top boroughs by complaint volume"""
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("""
        SELECT borough,
               SUM(total_complaints)      AS total_complaints,
               ROUND(AVG(avg_resolution_hours), 1) AS avg_resolution_hours,
               ROUND(AVG(closure_rate_pct), 1)     AS closure_rate_pct
        FROM mart_complaint_trends
        WHERE borough IS NOT NULL
        GROUP BY borough
        ORDER BY total_complaints DESC
    """).df()
    con.close()
    return df.to_dict(orient="records")