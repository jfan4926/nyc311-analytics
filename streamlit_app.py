import os
import re
import pickle
import shap
import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlparse
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DB_PATH = "data/processed/nyc311.duckdb"

# ── Input validation ──────────────────────────────────────
BLOCKED_PATTERNS = [
    r'\bdrop\b', r'\bdelete\b', r'\binsert\b', r'\bupdate\b',
    r'\btruncate\b', r'\bexec\b', r'\bexecute\b', r'\bunion\b',
    r'--', r';.*select', r'/\*'
]

def is_safe_input(text: str) -> tuple[bool, str]:
    text_lower = text.lower().strip()
    if len(text_lower) < 5:
        return False, "Question too short. Please ask something meaningful."
    if len(text_lower) > 200:
        return False, "Question too long. Please keep it under 200 characters."
    if not any(c.isalpha() for c in text_lower):
        return False, "Please ask a question in plain English."
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text_lower):
            return False, "Invalid input detected. Please ask a business question in plain English."
    return True, ""

def is_safe_sql(sql: str) -> tuple[bool, str]:
    sql_lower = sql.lower().strip()
    if not sql_lower.startswith('select'):
        return False, "Only SELECT queries are allowed."
    for pattern in [r'\bdrop\b', r'\bdelete\b', r'\binsert\b', r'\bupdate\b']:
        if re.search(pattern, sql_lower):
            return False, "Unsafe SQL detected."
    return True, ""

# ── RAG (keyword-based, no external dependencies) ────────
def retrieve_context(question: str, n_results: int = 3) -> str:
    q = question.lower()

    docs = {
        "trends": """Table mart_complaint_trends (already aggregated, one row per week+borough+complaint_type).
To get totals use SUM() with GROUP BY. ALWAYS include GROUP BY when using aggregate functions.
Columns: week, borough, complaint_type, total_complaints,
closed_complaints, avg_resolution_hours, closure_rate_pct.
Example - complaints by borough:
SELECT borough, SUM(total_complaints) AS total FROM mart_complaint_trends
WHERE borough IS NOT NULL GROUP BY borough ORDER BY total DESC""",

        "agency": """Table mart_agency_performance (one row per agency+complaint_type).
Columns: agency, agency_name, complaint_type, total_complaints,
avg_resolution_hours, p90_resolution_hours, pct_over_1_week.
Example - worst agency:
SELECT agency_name, ROUND(AVG(avg_resolution_hours),1) AS avg_hours
FROM mart_agency_performance GROUP BY agency_name ORDER BY avg_hours DESC LIMIT 5""",

        "features": """Table mart_ml_features (one row per complaint).
Columns: unique_key, created_at, resolution_hours, agency, complaint_type,
borough, channel_type, created_hour, created_dow, created_month,
is_weekend, hist_avg_resolution_hours, is_overdue (1=overdue, 0=on time).
Example - overdue rate by type:
SELECT complaint_type, ROUND(AVG(is_overdue)*100,1) AS overdue_pct
FROM mart_ml_features WHERE is_overdue IS NOT NULL
GROUP BY complaint_type HAVING COUNT(*)>=100 ORDER BY overdue_pct DESC LIMIT 10""",

        "example_volume": """Example - top complaint types by volume:
SELECT complaint_type, SUM(total_complaints) AS total
FROM mart_complaint_trends
GROUP BY complaint_type ORDER BY total DESC LIMIT 5""",

        "example_borough": """Example - closure rate by borough:
SELECT borough, ROUND(AVG(closure_rate_pct),1) AS avg_closure_rate
FROM mart_complaint_trends WHERE borough IS NOT NULL
GROUP BY borough ORDER BY avg_closure_rate DESC""",

        "example_agency": """Example - worst agency resolution time:
SELECT agency_name, ROUND(AVG(avg_resolution_hours),1) AS avg_hours
FROM mart_agency_performance
GROUP BY agency_name ORDER BY avg_hours DESC LIMIT 5""",

        "example_overdue": """Example - most overdue complaint types:
SELECT complaint_type, ROUND(AVG(is_overdue)*100,1) AS overdue_pct
FROM mart_ml_features WHERE is_overdue IS NOT NULL
GROUP BY complaint_type HAVING COUNT(*)>=100
ORDER BY overdue_pct DESC LIMIT 10""",

        "example_noise": """Example - borough with most noise complaints:
SELECT borough, SUM(total_complaints) AS total
FROM mart_complaint_trends
WHERE UPPER(complaint_type) LIKE '%NOISE%'
GROUP BY borough ORDER BY total DESC LIMIT 1""",

        "example_agency_count": """Example - agency with most complaints:
SELECT agency_name, SUM(total_complaints) AS total
FROM mart_agency_performance
GROUP BY agency_name ORDER BY total DESC LIMIT 5""",

        "example_week_trend": """Example - weekly complaint trend over time:
SELECT week, SUM(total_complaints) AS total
FROM mart_complaint_trends
GROUP BY week ORDER BY week""",

        "example_channel": """Example - complaints by submission channel:
SELECT channel_type, COUNT(*) AS total
FROM mart_ml_features
GROUP BY channel_type ORDER BY total DESC""",

        "example_resolution_type": """Example - slowest complaint types to resolve:
SELECT complaint_type, ROUND(AVG(avg_resolution_hours),1) AS avg_hours
FROM mart_agency_performance
GROUP BY complaint_type ORDER BY avg_hours DESC LIMIT 5""",
    }

    keywords = {
        "trends":               ["trend","week","volume","borough","closure","type","complaint","total"],
        "agency":               ["agency","department","performance","resolution","slow","fast","worst","best"],
        "features":             ["overdue","delay","predict","ml","feature","channel","hour","weekend"],
        "example_volume":       ["top","most","volume","count","popular","frequent"],
        "example_borough":      ["borough","closure","rate","brooklyn","manhattan","bronx","queens","staten"],
        "example_agency":       ["agency","worst","best","resolution","time","slow"],
        "example_overdue":      ["overdue","delay","late","risk","exceed","week"],
        "example_noise":        ["noise","sound","loud","residential","commercial","noise complaint"],
        "example_agency_count": ["agency","most","handle","complaints","count","total"],
        "example_week_trend":   ["trend","over time","weekly","month","change","grow"],
        "example_channel":      ["channel","online","phone","mobile","submit","how"],
        "example_resolution_type": ["slow","longest","takes","resolution","hours","type"],
    }

    scores = {k: sum(1 for kw in v if kw in q) for k, v in keywords.items()}
    scores["trends"] += 1  # always include schema

    top_ids  = sorted(scores, key=scores.get, reverse=True)[:n_results]
    selected = [docs[i] for i in top_ids]
    return "\n\n---\n\n".join(selected)

# ── Text-to-SQL ───────────────────────────────────────────
def query_database(sql: str) -> tuple[str, pd.DataFrame | None]:
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        df  = con.execute(sql).df()
        con.close()
        if df.empty:
            return "No results found.", None
        return df.to_string(index=False), df
    except Exception as e:
        return f"SQL Error: {e}", None

def ask(question: str, history: list = None) -> dict:
    if history is None:
        history = []
    llm     = ChatGroq(model="llama-3.1-8b-instant")
    context = retrieve_context(question)

    # 把历史放进system message
    history_context = ""
    for h in history[-3:]:
        history_context += f"\nPrevious Q: {h['question']}\nPrevious SQL: {h['sql']}\nPrevious result:\n{h.get('summary','')}\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a SQL expert for a NYC 311 complaints database.

Use the following relevant schema and examples to write accurate DuckDB SQL:

{context}

{f"Previous conversation context:{history_context}" if history_context else ""}

Rules:
- Return ONLY the SQL query, no explanation, no markdown, no backticks
- Use only the tables shown above
- Always use LIMIT not TOP
- For mart_complaint_trends: always use SUM() or AVG() with GROUP BY for totals
- For borough names use uppercase (BROOKLYN, QUEENS, MANHATTAN, BRONX, STATEN ISLAND)
- For aggregation queries no LIMIT needed unless specified
- If user uses pronouns like 'their', 'it', 'that', refer to previous result to identify the specific entity
"""),
        ("human", "{question}")
    ])

    sql            = (prompt | llm).invoke({"question": question}).content.strip()
    results_str, df = query_database(sql)
    return {
        "question": question,
        "sql":      sql,
        "results":  results_str,
        "data":     df.to_dict(orient="records") if df is not None else None,
    }

# ── Gauge ─────────────────────────────────────────────────
def make_gauge(proba: float, pred: int) -> str:
    pct       = round(float(proba) * 100, 1)
    label     = "HIGH RISK OF DELAY" if pred == 1 else "LIKELY ON TIME"
    icon      = "⚠" if pred == 1 else "✔"
    label_col = "#e74c3c" if pred == 1 else "#27ae60"
    total_len = 314.0
    fill_len  = total_len * float(proba)
    arc_color = "#27ae60" if pct < 30 else ("#f1c40f" if pct < 60 else "#e74c3c")
    return f"""<!DOCTYPE html>
<html><head><style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:transparent; display:flex; justify-content:center;
          align-items:center; height:210px; overflow:hidden; }}
  .wrap {{ position:relative; width:300px; height:170px; }}
  svg   {{ position:absolute; top:0; left:0; width:300px; height:170px; }}
  #arc-fill {{ stroke-dasharray: 0 {total_len:.1f};
               transition: stroke-dasharray 1.4s cubic-bezier(.17,.67,.35,1.2); }}
  .info {{ position:absolute; bottom:10px; left:0; right:0; text-align:center;
           font-family:'Segoe UI',system-ui,sans-serif; }}
  .pct  {{ font-size:38px; font-weight:800; color:{label_col}; line-height:1; }}
  .lbl  {{ font-size:12px; font-weight:600; color:{label_col};
           letter-spacing:.6px; margin-top:3px; }}
</style></head>
<body>
<div class="wrap">
  <svg viewBox="0 0 300 165">
    <path d="M 18 148 A 132 132 0 0 1 282 148"
          fill="none" stroke="#ecf0f1" stroke-width="20" stroke-linecap="round"/>
    <path id="arc-fill" d="M 18 148 A 132 132 0 0 1 282 148"
          fill="none" stroke="{arc_color}" stroke-width="20" stroke-linecap="round"
          pathLength="{total_len:.1f}"/>
  </svg>
  <div class="info">
    <div class="pct">{pct}%</div>
    <div class="lbl">{icon} {label}</div>
  </div>
</div>
<script>
  window.addEventListener('load', function() {{
    setTimeout(function() {{
      document.getElementById('arc-fill').style.strokeDasharray =
        '{fill_len:.1f} {total_len:.1f}';
    }}, 80);
  }});
</script>
</body></html>"""

# ── Model (cached) ────────────────────────────────────────
@st.cache_resource
def load_model_and_encoders():
    with open("ml_pipeline/models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("ml_pipeline/models/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_model_and_encoders()

FEATURE_COLS = [
    'agency', 'complaint_type', 'borough', 'channel_type',
    'created_hour', 'created_dow', 'created_month',
    'is_weekend', 'hist_avg_resolution_hours'
]
FEATURE_LABELS = {
    'agency':                    'Agency',
    'complaint_type':            'Complaint Type',
    'borough':                   'Borough',
    'channel_type':              'Channel',
    'created_hour':              'Hour of Day',
    'created_dow':               'Day of Week',
    'created_month':             'Month',
    'is_weekend':                'Is Weekend',
    'hist_avg_resolution_hours': 'Historical Avg Resolution (hrs)'
}

def safe_encode(encoder, value):
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return 0

# ── Page config ───────────────────────────────────────────
st.set_page_config(page_title="NYC 311 Analytics", page_icon="🗽", layout="wide")

st.markdown("""
<h1 style='text-align:center;color:#1f77b4;'>🗽 NYC 311 Complaint Analytics</h1>
<p style='text-align:center;color:gray;font-size:16px;'>
    dbt · XGBoost · SHAP · LLaMA 3.1 · LangChain · FastAPI
</p>
""", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; max-width:700px; margin:0 auto; color:#555; font-size:15px; line-height:1.7;'>
This platform analyzes <strong>499K+ NYC 311 service requests</strong> from 2022–2026, 
covering noise complaints, illegal parking, housing issues, and more across all five boroughs. 
<br>Use the tabs below to explore trends, ask questions in plain English, or predict whether a 
complaint will be resolved on time.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

#st.divider()
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] {
    gap: 16px;
}
.stTabs [data-baseweb="tab"] {
    font-size: 25px;
    padding: 8px 20px;
}
</style>
""", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Ask the Data", "🔮 Predict & Explain"])

# ════════════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        text-align: center !important;
    }
    [data-testid="stMetricLabel"] {
        width: 100% !important;
        text-align: center !important;
        display: block !important;
    }
    </style>
    """, unsafe_allow_html=True)
    con  = duckdb.connect(DB_PATH, read_only=True)

    st.markdown("""
    <div style='background:#f0f7ff; border-left:4px solid #1f77b4; 
                border-radius:6px; padding:12px 16px; margin-bottom:16px; font-size:13px; color:#444;'>
    <strong>📊 About this dashboard</strong> · 
    NYC 311 service requests (499K+, Apr 2022–May 2026) · 
    Tracks complaint volume, agency SLA performance, borough trends, and 4-month Prophet forecast · 
    <a href='https://data.cityofnewyork.us/resource/erm2-nwe9' target='_blank' style='color:#1f77b4;'>Data source ↗</a>
    </div>
    """, unsafe_allow_html=True)
    kpis = con.execute("""
        SELECT SUM(total_complaints)               AS total_complaints,
               ROUND(AVG(avg_resolution_hours), 1) AS avg_resolution_hours,
               ROUND(AVG(closure_rate_pct), 1)     AS avg_closure_rate
        FROM mart_complaint_trends
    """).df()

    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric("🗂 Total Complaints", f"{int(kpis['total_complaints'][0]):,}")
    c2.metric("⏱ Avg Resolution",    f"{kpis['avg_resolution_hours'][0]}h")
    c3.metric("✅ Closure Rate",      f"{kpis['avg_closure_rate'][0]}%")
    c4.metric("📍 Boroughs",          "5")
    components.html("""
    <script>
    setTimeout(() => {
        const metrics = window.parent.document.querySelectorAll('[data-testid="stMetricValue"]');
        metrics.forEach(el => {
            const text = el.innerText.replace(/[^0-9.]/g, '');
            const target = parseFloat(text);
            if (isNaN(target)) return;
            const suffix = el.innerText.replace(/[0-9.,]/g, '');
            const isFloat = el.innerText.includes('.');
            let current = 0;
            const steps = 50;
            const increment = target / steps;
            el.innerText = isFloat ? '0' + suffix : '0';
            const timer = setInterval(() => {
                current += increment;
                if (current >= target) { current = target; clearInterval(timer); }
                el.innerText = isFloat
                    ? current.toFixed(1) + suffix
                    : Math.floor(current).toLocaleString() + suffix;
            }, 1200 / steps);
        });
    }, 200);
    </script>
    """, height=0)
    st.divider()

    cl, cr = st.columns(2)
    with cl:
        st.subheader("📍 Complaints by Borough")
        df = con.execute("""
            SELECT borough, SUM(total_complaints) AS total_complaints
            FROM mart_complaint_trends WHERE borough IS NOT NULL
            GROUP BY borough ORDER BY total_complaints DESC
        """).df()
        fig = px.bar(df, x="borough", y="total_complaints",
                     color="total_complaints", color_continuous_scale="Blues",
                     text="total_complaints")
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(coloraxis_showscale=False, showlegend=False,
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.subheader("⏱ Avg Resolution Time by Borough")
        df2 = con.execute("""
            SELECT borough, ROUND(AVG(avg_resolution_hours),1) AS avg_hours
            FROM mart_complaint_trends WHERE borough IS NOT NULL
            GROUP BY borough ORDER BY avg_hours DESC
        """).df()
        fig2 = px.bar(df2, x="borough", y="avg_hours",
                      color="avg_hours", color_continuous_scale="Reds",
                      text="avg_hours")
        fig2.update_traces(texttemplate='%{text}h', textposition='outside')
        fig2.update_layout(coloraxis_showscale=False, showlegend=False,
                           plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📈 Weekly Complaint Volume")
    df3 = con.execute("""
        SELECT week, SUM(total_complaints) AS total_complaints
        FROM mart_complaint_trends GROUP BY week ORDER BY week
    """).df()
    fig3 = px.area(df3, x="week", y="total_complaints",
                   color_discrete_sequence=["#1f77b4"],
                   labels={"total_complaints":"Complaints","week":"Week"})
    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig3.update_traces(fillcolor='rgba(31,119,180,0.15)')
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("🏆 Agency Performance")
    df4 = con.execute("""
        SELECT agency_name,
               ROUND(AVG(avg_resolution_hours),1) AS avg_hours,
               ROUND(AVG(pct_over_1_week),1)      AS pct_over_1_week
        FROM mart_agency_performance
        GROUP BY agency_name ORDER BY avg_hours DESC LIMIT 10
    """).df()
    fig4 = px.scatter(df4, x="avg_hours", y="pct_over_1_week",
                      text="agency_name", size="avg_hours",
                      color="pct_over_1_week", color_continuous_scale="RdYlGn_r",
                      labels={"avg_hours":"Avg Resolution (hrs)",
                              "pct_over_1_week":"% Over 1 Week"})
    fig4.update_traces(textposition='top center')
    fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig4, use_container_width=True)

    
    # ── Prophet Forecast ──────────────────────────────────
    st.subheader("📈 4-Week Complaint Volume Forecast")
    st.caption("Prophet time series model · 77 months training data (2020–2026) · yearly seasonality · 95% confidence interval")
    
    @st.cache_data
    def run_forecast():
        from prophet import Prophet
        df_hist = pd.read_csv("data/forecasting/monthly_complaints.csv")
        df_hist['ds'] = pd.to_datetime(df_hist['ds'])

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        m.fit(df_hist)

        future   = m.make_future_dataframe(periods=4, freq='MS')
        forecast = m.predict(future)
        return df_hist, forecast

    df_hist, forecast = run_forecast()

    # 合并历史+预测
    hist_plot = df_hist.rename(columns={"ds": "date", "y": "complaints"})
    hist_plot["type"] = "Historical"

    pred_plot = forecast[forecast['ds'] > df_hist['ds'].max()][['ds','yhat','yhat_lower','yhat_upper']].copy()
    pred_plot.columns = ['date','complaints','lower','upper']
    pred_plot["type"] = "Forecast"

    fig_forecast = go.Figure()

    # 历史线
    fig_forecast.add_trace(go.Scatter(
        x=hist_plot['date'], y=hist_plot['complaints'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4)
    ))

    # 预测区间
    fig_forecast.add_trace(go.Scatter(
        x=pd.concat([pred_plot['date'], pred_plot['date'][::-1]]),
        y=pd.concat([pred_plot['upper'], pred_plot['lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(255,127,14,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval',
        showlegend=True
    ))

    # 预测线
    fig_forecast.add_trace(go.Scatter(
        x=pred_plot['date'], y=pred_plot['complaints'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))

    fig_forecast.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        legend=dict(orientation='h', y=1.1),
        xaxis_title="Month",
        yaxis_title="Complaints",
        hovermode='x unified'
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    # 预测数字展示
    forecast_months = pred_plot.sort_values('date')
    fc1, fc2, fc3, fc4 = st.columns(4)
    for i, (col, row) in enumerate(zip([fc1,fc2,fc3,fc4], forecast_months.itertuples())):
        col.metric(
            label=row.date.strftime("%b %Y"),
            value=f"{int(row.complaints):,}",
            delta=f"±{int((row.upper-row.lower)/2):,}"
        )


    #── Heatmap ──────────────────────────────────
    st.subheader("🗺 Complaint Density Heatmap")
    st.caption("Geographic distribution of complaints — darker = higher density")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_borough = st.selectbox(
            "Filter by Borough",
            ["All"] + ["BRONX","BROOKLYN","MANHATTAN","QUEENS","STATEN ISLAND"],
            key="map_borough"
        )
    with col_f2:
        selected_type = st.selectbox(
            "Filter by Complaint Type",
            ["All", "ILLEGAL PARKING", "NOISE - RESIDENTIAL",
             "BLOCKED DRIVEWAY", "HEAT/HOT WATER", "STREET CONDITION",
             "RODENT", "UNSANITARY CONDITION", "ABANDONED VEHICLE"],
            key="map_type"
        )

    borough_filter = f"AND UPPER(TRIM(borough)) = '{selected_borough}'" if selected_borough != "All" else ""
    type_filter = f"AND UPPER(TRIM(complaint_type)) = '{selected_type}'" if selected_type != "All" else ""

    BOROUGH_CENTERS = {
        "BRONX":         {"lat": 40.8448, "lon": -73.8648},
        "BROOKLYN":      {"lat": 40.6782, "lon": -73.9442},
        "MANHATTAN":     {"lat": 40.7831, "lon": -73.9712},
        "QUEENS":        {"lat": 40.7282, "lon": -73.7949},
        "STATEN ISLAND": {"lat": 40.5795, "lon": -74.1502},
        "All":           {"lat": 40.7128, "lon": -74.0060},
    }

    map_center = BOROUGH_CENTERS[selected_borough]
    map_zoom   = 10 if selected_borough == "All" else 12
    map_df = con.execute(f"""
        SELECT 
            TRY_CAST(latitude AS DOUBLE) AS latitude,
            TRY_CAST(longitude AS DOUBLE) AS longitude,
            UPPER(TRIM(complaint_type)) AS complaint_type,
            CASE WHEN borough = 'Unspecified' THEN NULL 
                 ELSE UPPER(TRIM(borough)) END AS borough
        FROM read_parquet('data/raw/nyc311.parquet')
        WHERE latitude IS NOT NULL
          AND longitude IS NOT NULL
          AND TRY_CAST(latitude AS DOUBLE) BETWEEN 40.4 AND 41.0
          AND TRY_CAST(longitude AS DOUBLE) BETWEEN -74.3 AND -73.7
          {borough_filter}
          {type_filter}
        ORDER BY RANDOM()
        LIMIT 30000
    """).df()

    if selected_type == "All":
        color_scale = "Blues"
    else:
        color_scale = "Reds"

    fig_map = px.density_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        center=map_center,
        zoom=map_zoom,
        radius=5,
        # center={"lat": 40.7128, "lon": -74.0060},
        # zoom=10,
        mapbox_style="carto-positron",
        color_continuous_scale=color_scale,
        opacity=0.7,
        hover_data=["complaint_type", "borough"],
        labels={"z": "Density"}
    )
    fig_map.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_map, use_container_width=True)

    con.close()

# ════════════════════════════════════════════════════════
# TAB 2: ASK THE DATA
# ════════════════════════════════════════════════════════

with tab2:
    st.markdown("""
    <div style='background:#f0fff4; border-left:4px solid #27ae60; 
                border-radius:6px; padding:12px 16px; margin-bottom:16px; font-size:13px; color:#444;'>
    <strong>🤖 How it works</strong> · 
    Type a question in plain English → LLaMA 3.1 generates SQL → DuckDB executes → results auto-visualized · 
    Supports follow-up questions ("What about Brooklyn?") · 
    Powered by Groq API + LangChain + RAG
    </div>
    """, unsafe_allow_html=True)
    st.subheader("💬 Ask questions in plain English")
    st.caption("LLaMA 3.1 generates SQL → DuckDB executes → Results visualized automatically")

    col_ex1, col_ex2, col_ex3 = st.columns(3)
    if col_ex1.button("🏙 Highest closure rate by borough?"):
        st.session_state.question = "Which borough has the highest closure rate?"
    if col_ex2.button("📋 Top 5 complaint types by volume?"):
        st.session_state.question = "What is the top complaint type by total volume?"
    if col_ex3.button("🏢 Worst agency resolution time?"):
        st.session_state.question = "Which agency has the worst average resolution time?"

    col_ex4, col_ex5, col_ex6 = st.columns(3)
    if col_ex4.button("🔊 Which borough has most noise complaints?"):
        st.session_state.question = "Which borough has the most NOISE - RESIDENTIAL complaints by total volume?"
    if col_ex5.button("⏱ Slowest complaint types to resolve?"):
        st.session_state.question = "What is the top complaint type by average resolution time?"
    if col_ex6.button("🏆 Which agency handles most complaints?"):
        st.session_state.question = "Which agency has the most total complaints?"

    #chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    question = st.text_input(
        "Your question:",
        value=st.session_state.get("question", ""),
        placeholder="e.g. Which borough has the most noise complaints?"
    )

    if st.button("🔍 Ask", type="primary") and question:
        is_safe, err_msg = is_safe_input(question)
        if not is_safe:
            st.error(f"⚠️ {err_msg}")
            st.stop()

        st.markdown("### 🧠 Agent Process")
        cs1, cs2, cs3 = st.columns(3)

        with cs1:
            with st.status("Step 1: Understanding question...", expanded=True) as s1:
                st.write(f"**Input:** {question}")
                st.write("Sending to LLaMA 3.1 via Groq API...")
                s1.update(label="✅ Step 1: Question understood",
                          state="complete", expanded=True)

        result = None
        with cs2:
            with st.status("Step 2: Generating SQL...", expanded=True) as s2:
                try:
                    result = ask(question, st.session_state.chat_history)

                    sql    = result["sql"]

                    sql_safe, sql_err = is_safe_sql(sql)
                    if not sql_safe:
                        s2.update(label="❌ Step 2: Unsafe SQL",
                                  state="error", expanded=True)
                        st.error(sql_err)
                        st.stop()

                    sql_fmt = sqlparse.format(
                        sql, reindent=True,
                        keyword_case='upper', indent_width=4)
                    st.code(sql_fmt, language="sql")
                    # st.code(sql, language="sql")  # unformatted fallback
                    s2.update(label="✅ Step 2: SQL generated",
                              state="complete", expanded=True)
                except Exception as e:
                    s2.update(label="❌ Step 2: Failed",
                              state="error", expanded=True)
                    st.error(f"Error: {str(e)}")
                    st.stop()

        with cs3:
            with st.status("Step 3: Querying DuckDB...", expanded=True) as s3:
                st.write("Executing against mart tables...")
                if result and result.get("data"):
                    st.write(f"✓ Retrieved **{len(result['data'])}** rows")
                s3.update(label="✅ Step 3: Results ready",
                          state="complete", expanded=True)
        if result and result.get("sql"):
            # let LLM know the previous result
            summary = ""
            if result.get("data") and len(result["data"]) > 0:
                df_tmp = pd.DataFrame(result["data"])
                summary = df_tmp.head(3).to_string(index=False)
            st.session_state.chat_history.append({
                "question": question,
                "sql": result["sql"],
                "summary": summary
            })

        st.divider()

        if result and result.get("data"):
            df_result  = pd.DataFrame(result["data"])
            num_cols   = df_result.select_dtypes(include='number').columns.tolist()
            str_cols   = df_result.select_dtypes(exclude='number').columns.tolist()
            tbl_height = min(40 + len(df_result) * 35, 400)

            ct, cc = st.columns(2)
            with ct:
                st.markdown("### 📋 Results")
                st.dataframe(df_result, use_container_width=True, height=tbl_height)

            with cc:
                st.markdown("### 📊 Auto Visualization")
                if num_cols and str_cols and len(df_result) > 1:
                    fig = px.bar(df_result, x=str_cols[0], y=num_cols[0],
                                 color=num_cols[0], color_continuous_scale="Blues",
                                 text=num_cols[0])
                    fig.update_traces(texttemplate='%{text:,.0f}',
                                      textposition='outside')
                    fig.update_layout(coloraxis_showscale=False,
                                      plot_bgcolor='rgba(0,0,0,0)',
                                      xaxis_tickangle=-30)
                    st.plotly_chart(fig, use_container_width=True)
                elif num_cols and str_cols and len(df_result) == 1:
                    for col in num_cols:
                        st.metric(col, f"{df_result[col][0]:,.1f}")
                else:
                    st.dataframe(df_result)
        elif result:
            st.text(result.get("results", "No results"))
            # ── 对话历史 ──
        if len(st.session_state.chat_history) > 1:
            with st.expander(f"💬 Conversation History ({len(st.session_state.chat_history)} questions)"):
                for i, h in enumerate(st.session_state.chat_history):
                    st.markdown(f"**Q{i+1}:** {h['question']}")
                    st.code(h['sql'], language="sql")
                    st.divider()
            if st.button("🗑 Clear History"):
                st.session_state.chat_history = []
                st.rerun()


# ════════════════════════════════════════════════════════
# TAB 3: PREDICT & EXPLAIN
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style='background:#fff8f0; border-left:4px solid #e67e22; 
                border-radius:6px; padding:12px 16px; margin-bottom:16px; font-size:13px; color:#444;'>
    <strong>🔮 How it works</strong> · 
    XGBoost classifier (AUC=0.96) trained on 330K complaints · 
    Predicts whether a complaint will exceed 1-week resolution · 
    SHAP values explain each feature's contribution · 
    Similar historical cases provide real-world context
    </div>
    """, unsafe_allow_html=True)
    st.subheader("🔮 Will this complaint be resolved on time?")
    st.caption("XGBoost · AUC = 0.96 · Explained with SHAP")

    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.markdown("### ⚙️ Input Parameters")
        agency = st.selectbox("Agency", [
            'DCWP','DEP','DHS','DOB','DOHMH',
            'DOT','DPR','DSNY','HPD','NYPD'
        ])
        complaint_type = st.selectbox("Complaint Type", [
            'CURB CONDITION','STREET LIGHT CONDITION',
            'STREET SIGN - DANGLING','RODENT','GENERAL',
            'ABANDONED VEHICLE','BLOCKED DRIVEWAY',
            'HEAT/HOT WATER','ILLEGAL PARKING',
            'NOISE - RESIDENTIAL','STREET CONDITION',
            'UNSANITARY CONDITION'
        ])
        borough      = st.selectbox("Borough", [
            'BRONX','BROOKLYN','MANHATTAN','QUEENS','STATEN ISLAND'
        ])
        channel_type = st.selectbox("Channel", [
            'MOBILE','ONLINE','PHONE','UNKNOWN'
        ])
        created_hour  = st.slider("Hour of Day",         0, 23, 12)
        created_month = st.slider("Month",                1, 12,  6)
        created_dow   = st.slider("Day of Week (0=Sun)", 0,  6,  1)
        is_weekend    = created_dow in [0, 6]
        st.info(f"Weekend: {'Yes ✅' if is_weekend else 'No'}")
        predict_btn   = st.button("🔮 Predict & Explain", type="primary")

    with col_out:
        if predict_btn:
            with st.spinner("Running model + SHAP analysis..."):

                # ── Build features ──
                con      = duckdb.connect(DB_PATH, read_only=True)
                hist_row = con.execute("""
                    SELECT COALESCE(hist_avg_resolution_hours, 0)
                    FROM mart_ml_features
                    WHERE agency = ? AND complaint_type = ?
                    LIMIT 1
                """, [agency, complaint_type]).fetchone()
                con.close()
                hist_avg = hist_row[0] if hist_row else 0.0

                features = pd.DataFrame([{
                    "agency":                    safe_encode(encoders['agency'],         agency),
                    "complaint_type":            safe_encode(encoders['complaint_type'], complaint_type),
                    "borough":                   safe_encode(encoders['borough'],        borough),
                    "channel_type":              safe_encode(encoders['channel_type'],   channel_type),
                    "created_hour":              created_hour,
                    "created_dow":               created_dow,
                    "created_month":             created_month,
                    "is_weekend":                int(is_weekend),
                    "hist_avg_resolution_hours": hist_avg
                }])

                proba = model.predict_proba(features)[0][1]
                pred  = int(proba >= 0.5)
                pct   = int(proba * 100)

                # ── Row 1: Gauge + Similar Cases ──────────────────────
                r1_left, r1_right = st.columns(2)

                with r1_left:
                    st.markdown("### 🎯 Prediction Result")
                    components.html(make_gauge(proba, pred), height=220)
                    if pct < 30:
                        st.success("🟢 LOW RISK — Likely resolved on time")
                    elif pct < 60:
                        st.warning("🟡 MEDIUM RISK — Some chance of delay")
                    else:
                        st.error("🔴 HIGH RISK — Very likely to be delayed")

                with r1_right:
                    st.markdown("### 🔎 Similar Historical Cases")
                    st.caption("Complaints with same agency · type · borough")
                    con = duckdb.connect(DB_PATH, read_only=True)
                    similar_df = con.execute(f"""
                        SELECT
                            created_at::DATE                                    AS date,
                            ROUND(resolution_hours, 1)                          AS res_hours,
                            channel_type,
                            CASE WHEN is_overdue = 1 THEN '⚠️ Overdue'
                                 ELSE '✅ On Time' END                          AS status
                        FROM mart_ml_features
                        WHERE agency         = '{agency}'
                          AND complaint_type = '{complaint_type}'
                          AND borough        = '{borough}'
                          AND resolution_hours IS NOT NULL
                        ORDER BY ABS(created_hour - {created_hour})
                        LIMIT 5
                    """).df()
                    con.close()
                    if len(similar_df) > 0:
                        avg_sim = similar_df['res_hours'].mean()
                        st.metric("Avg Resolution (similar cases)", f"{avg_sim:.1f}h")
                        st.dataframe(similar_df, use_container_width=True,
                                     hide_index=True)
                    else:
                        st.info("No historical cases found with these parameters.")

                st.divider()

                # ── Row 2: SHAP + Feature Importance ──────────────────
                r2_left, r2_right = st.columns(2)

                explainer   = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(features)

                shap_df = pd.DataFrame({
                    'feature':       [FEATURE_LABELS[c] for c in FEATURE_COLS],
                    'shap_value':    shap_values[0],
                    'feature_value': [
                        agency, complaint_type, borough, channel_type,
                        created_hour, created_dow, created_month,
                        is_weekend, round(hist_avg, 1)
                    ]
                }).sort_values('shap_value', key=abs, ascending=True)

                shap_df['color'] = shap_df['shap_value'].apply(
                    lambda x: '#e74c3c' if x > 0 else '#27ae60')
                shap_df['label'] = shap_df.apply(
                    lambda r: f"{r['feature']} = {r['feature_value']}", axis=1)

                with r2_left:
                    st.markdown("### 🔍 Why this prediction?")
                    st.caption("SHAP values: each feature's contribution")
                    fig_shap = go.Figure(go.Bar(
                        x=shap_df['shap_value'],
                        y=shap_df['label'],
                        orientation='h',
                        marker_color=shap_df['color'],
                        text=[f"{v:.4f}" for v in shap_df['shap_value']],
                        textposition='outside'
                    ))
                    fig_shap.update_layout(
                        title="🔴 increases risk  🟢 reduces risk",
                        xaxis_title="SHAP Value",
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=380, margin=dict(l=10, r=30, t=40, b=20)
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)

                with r2_right:
                    st.markdown("### 📊 Overall Feature Importance")
                    st.caption("Based on all training data")
                    imp_df = pd.DataFrame({
                        'feature':    [FEATURE_LABELS[c] for c in FEATURE_COLS],
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=True)
                    fig_imp = px.bar(
                        imp_df, x='importance', y='feature',
                        orientation='h', color='importance',
                        color_continuous_scale='Blues',
                        text=[f"{v:.4f}" for v in imp_df['importance']]
                    )
                    fig_imp.update_layout(
                        coloraxis_showscale=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=380, margin=dict(l=10, r=10, t=10, b=20)
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

        else:
            st.markdown("""
            <div style='text-align:center;padding:80px;color:gray;'>
              <h3>👈 Fill in the parameters and click Predict</h3>
              <p>The model will explain its decision using SHAP values</p>
            </div>
            """, unsafe_allow_html=True)


st.divider()
st.markdown("""
<div style='text-align:center; color:#aaa; font-size:12px; padding:16px 0 8px 0;'>
    <p>
        Data source: <a href='https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9' 
        target='_blank' style='color:#aaa;'>NYC Open Data — 311 Service Requests</a> · 
        Updated through May 2026
    </p>
    <p>
        ⚠️ This platform is intended for analytical and educational purposes only. 
        Predictions are probabilistic and should not be used for official decision-making.
    </p>
    <p>
        <a href='https://github.com/jfan4926/nyc311-analytics' target='_blank' 
        style='color:#aaa;'>📁 View on GitHub</a> · 
        Built with Python · dbt · XGBoost · LLaMA 3.1 · Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
        # ── Back to top ───────────────────────────────────────
st.markdown("""
<a href="#nyc-311-complaint-analytics" style="
    position: fixed;
    bottom: 32px;
    right: 32px;
    width: 42px;
    height: 42px;
    border-radius: 50%;
    background: #1f77b4;
    color: white;
    border: none;
    font-size: 20px;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    text-decoration: none;
    line-height: 1;
">↑</a>
""", unsafe_allow_html=True)
