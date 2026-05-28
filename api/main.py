import duckdb
import mlflow
import mlflow.sklearn
import pandas as pd
import pickle
import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from api.llm.text_to_sql import ask
from dotenv import load_dotenv
from fastapi.security import APIKeyHeader
from fastapi import Security
load_dotenv()

DB_PATH = "data/processed/nyc311.duckdb"

limiter = Limiter(key_func=get_remote_address)

API_KEY = os.getenv("INTERNAL_API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: str = Security(api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    return key

app = FastAPI(
    title="NYC311 Analytics API",
    description="Text-to-SQL and complaint overdue prediction",
    version="1.0.0"
)

app.state.limiter = limiter

# ── 启动时加载模型 ─────────────────────────────────────────
@app.on_event("startup")
async def load_model():
    global model, encoders
    try:
        with open("ml_pipeline/models/model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("ml_pipeline/models/encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        print("Model and encoders loaded from pickle")
    except Exception as e:
        print(f"Warning: Model loading failed: {e}")
        model = None
        encoders = None
# @app.on_event("startup")
# async def load_model():
#     global model, encoders
#     try:
#         runs = mlflow.search_runs(experiment_names=["nyc311-overdue-prediction"])
#         best_run = runs.sort_values("metrics.test_auc", ascending=False).iloc[0]
#         run_id = best_run["run_id"]
#         model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
#         with open("ml_pipeline/models/encoders.pkl", "rb") as f:
#             encoders = pickle.load(f)
#         print(f"Model loaded: {run_id}")
#     except Exception as e:
#         print(f"Warning: Model loading failed: {e}")
#         model = None
#         encoders = None

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

# @app.post("/query")
# @limiter.limit("10/minute")
# def query(request: Request, req: QuestionRequest):
#     """Natural language → SQL → results"""
#     try:
#         result = ask(req.question)
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
@limiter.limit("10/minute")
def query(request: Request, req: QuestionRequest,
          _: str = Security(verify_api_key)):
    try:
        result = ask(req.question)
        # 尝试把结果转成JSON
        try:
            con = duckdb.connect(DB_PATH, read_only=True)
            df = con.execute(result["sql"]).df()
            con.close()
            result["data"] = df.to_dict(orient="records")
            result["columns"] = df.columns.tolist()
        except Exception:
            result["data"] = None
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(req: PredictRequest, _: str = Security(verify_api_key)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
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

        def safe_encode(encoder, value):
            if value in encoder.classes_:
                return int(encoder.transform([value])[0])
            return 0

        features = pd.DataFrame([{
            "agency":                    safe_encode(encoders['agency'], req.agency),
            "complaint_type":            safe_encode(encoders['complaint_type'], req.complaint_type),
            "borough":                   safe_encode(encoders['borough'], req.borough),
            "channel_type":              safe_encode(encoders['channel_type'], req.channel_type),
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
               SUM(total_complaints)                     AS total_complaints,
               ROUND(AVG(avg_resolution_hours), 1)       AS avg_resolution_hours,
               ROUND(AVG(closure_rate_pct), 1)           AS closure_rate_pct
        FROM mart_complaint_trends
        WHERE borough IS NOT NULL
        GROUP BY borough
        ORDER BY total_complaints DESC
    """).df()
    con.close()
    return df.to_dict(orient="records")



