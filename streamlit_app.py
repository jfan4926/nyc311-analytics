import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import shap
import pickle
import mlflow
import mlflow.sklearn
import numpy as np
from dotenv import load_dotenv
import sqlparse
import re
import streamlit.components.v1 as components
import os
# ── Input validation ──────────────────────────────────────

INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")
HEADERS = {"X-API-Key": INTERNAL_API_KEY}

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

# ── Gauge (arc fill, no needle, animated) ────────────────
def make_gauge(proba: float, pred: int) -> str:
    pct       = round(float(proba) * 100, 1)
    label     = "HIGH RISK OF DELAY" if pred == 1 else "LIKELY ON TIME"
    icon      = "⚠" if pred == 1 else "✔"
    label_col = "#e74c3c" if pred == 1 else "#27ae60"
    total_len = 314.0          # π × r=100, semicircle pathLength
    fill_len  = total_len * float(proba)

    if pct < 30:
        arc_color = "#27ae60"
    elif pct < 60:
        arc_color = "#f1c40f"
    else:
        arc_color = "#e74c3c"

    return f"""<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{
    background:transparent;
    display:flex; justify-content:center; align-items:center;
    height:210px; overflow:hidden;
  }}
  .wrap {{ position:relative; width:300px; height:170px; }}
  svg   {{ position:absolute; top:0; left:0; width:300px; height:170px; }}
  #arc-fill {{
    stroke-dasharray: 0 {total_len:.1f};
    transition: stroke-dasharray 1.4s cubic-bezier(.17,.67,.35,1.2);
  }}
  .info {{
    position:absolute;
    bottom:10px; left:0; right:0;
    text-align:center;
    font-family:'Segoe UI',system-ui,sans-serif;
  }}
  .pct  {{ font-size:38px; font-weight:800; color:{label_col}; line-height:1; }}
  .lbl  {{ font-size:12px; font-weight:600; color:{label_col};
           letter-spacing:.6px; margin-top:3px; }}
  .zone {{ font-size:10px; font-weight:600; fill:#bbb; }}
</style>
</head>
<body>
<div class="wrap">
  <svg viewBox="0 0 300 165">
    <!-- background track -->
    <path d="M 18 148 A 132 132 0 0 1 282 148"
          fill="none" stroke="#ecf0f1"
          stroke-width="20" stroke-linecap="round"/>
    <!-- animated fill -->
    <path id="arc-fill"
          d="M 18 148 A 132 132 0 0 1 282 148"
          fill="none" stroke="{arc_color}"
          stroke-width="20" stroke-linecap="round"
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
</body>
</html>"""

# ── Config ────────────────────────────────────────────────
load_dotenv()
DB_PATH = "data/processed/nyc311.duckdb"
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="NYC 311 Analytics", page_icon="🗽", layout="wide")

# ── Load model (cached) ───────────────────────────────────
@st.cache_resource
def load_model_and_encoders():
    runs     = mlflow.search_runs(experiment_names=["nyc311-overdue-prediction"])
    best_run = runs.sort_values("metrics.test_auc", ascending=False).iloc[0]
    run_id   = best_run["run_id"]
    model    = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
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

# ── Header ────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center;color:#1f77b4;'>🗽 NYC 311 Complaint Analytics</h1>
<p style='text-align:center;color:gray;font-size:16px;'>
    dbt · XGBoost · SHAP · LLaMA 3.1 · LangChain · FastAPI
</p>
""", unsafe_allow_html=True)
st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Ask the Data", "🔮 Predict & Explain"])

# ════════════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ════════════════════════════════════════════════════════
with tab1:
    con  = duckdb.connect(DB_PATH, read_only=True)
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
    con.close()

# ════════════════════════════════════════════════════════
# TAB 2: ASK THE DATA
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("💬 Ask questions in plain English")
    st.caption("LLaMA 3.1 generates SQL → DuckDB executes → Results visualized automatically")

    col_ex1, col_ex2, col_ex3 = st.columns(3)
    if col_ex1.button("🏙 Highest closure rate by borough?"):
        st.session_state.question = "Which borough has the highest closure rate?"
    if col_ex2.button("📋 Top 5 complaint types by volume?"):
        st.session_state.question = "What are the top 5 complaint types by total volume?"
    if col_ex3.button("🏢 Worst agency resolution time?"):
        st.session_state.question = "Which agency has the worst average resolution time?"

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

        with cs2:
            with st.status("Step 2: Generating SQL...", expanded=True) as s2:
                try:
                    resp   = requests.post(f"{API_URL}/query",
                                           json={"question": question},
                                            headers=HEADERS)
                    result = resp.json()
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
                    st.error(f"API error: {str(e)}")
                    st.stop()

        with cs3:
            with st.status("Step 3: Querying DuckDB...", expanded=True) as s3:
                st.write("Executing against mart tables...")
                if result.get("data"):
                    st.write(f"✓ Retrieved **{len(result['data'])}** rows")
                s3.update(label="✅ Step 3: Results ready",
                          state="complete", expanded=True)

        st.divider()

        if result.get("data"):
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
        else:
            st.text(result.get("results", "No results"))

        # # Legacy text-parsing fallback (kept for reference)
        # results_text = result["results"]
        # try:
        #     lines = [l for l in results_text.strip().split('\n') if l.strip()]
        #     if len(lines) >= 2:
        #         headers = lines[0].split()
        #         rows = [l.split() for l in lines[1:]]
        #         max_cols = len(headers)
        #         rows = [r[:max_cols] for r in rows if len(r) >= max_cols]
        #         df_result = pd.DataFrame(rows, columns=headers)
        #         for col in df_result.columns:
        #             try: df_result[col] = pd.to_numeric(df_result[col])
        #             except: pass
        #         st.dataframe(df_result)
        #     else:
        #         st.text(results_text)
        # except:
        #     st.text(results_text)

# ════════════════════════════════════════════════════════
# TAB 3: PREDICT & EXPLAIN
# ════════════════════════════════════════════════════════
with tab3:
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

                # ── Gauge ──
                st.markdown("### 🎯 Prediction Result")
                components.html(make_gauge(proba, pred), height=220)

                # ── Risk badge ──
                if pct < 30:
                    st.success("🟢 LOW RISK — Complaint likely resolved on time")
                elif pct < 60:
                    st.warning("🟡 MEDIUM RISK — Some chance of delay")
                else:
                    st.error("🔴 HIGH RISK — Complaint very likely to be delayed")

                st.divider()

                # ── SHAP waterfall ──
                st.markdown("### 🔍 Why did the model predict this?")
                st.caption("SHAP values: each feature's contribution to the prediction")

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

                fig_shap = go.Figure(go.Bar(
                    x=shap_df['shap_value'],
                    y=shap_df['label'],
                    orientation='h',
                    marker_color=shap_df['color'],
                    text=[f"{v:.4f}" for v in shap_df['shap_value']],
                    textposition='outside'
                ))
                fig_shap.update_layout(
                    title="🔴 Red = increases delay risk   🟢 Green = reduces delay risk",
                    xaxis_title="SHAP Value (impact on prediction)",
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400, margin=dict(l=20, r=20)
                )
                st.plotly_chart(fig_shap, use_container_width=True)

                # ── Overall feature importance ──
                st.markdown("### 📊 Overall Model Feature Importance")
                st.caption("Based on all training data, not just this prediction")

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
                    height=350
                )
                st.plotly_chart(fig_imp, use_container_width=True)

        else:
            st.markdown("""
            <div style='text-align:center;padding:80px;color:gray;'>
              <h3>👈 Fill in the parameters and click Predict</h3>
              <p>The model will explain its decision using SHAP values</p>
            </div>
            """, unsafe_allow_html=True)