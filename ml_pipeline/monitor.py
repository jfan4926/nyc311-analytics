import duckdb
import pandas as pd
from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently import Report

DB_PATH = "data/processed/nyc311.duckdb"

def load_data():
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("""
        SELECT
            is_overdue,
            agency,
            complaint_type,
            borough,
            channel_type,
            created_hour,
            created_dow,
            created_month,
            is_weekend,
            hist_avg_resolution_hours,
            created_at
        FROM mart_ml_features
        WHERE is_overdue IS NOT NULL
    """).df()
    con.close()
    return df

def run_drift_report():
    df = load_data()

    # 按时间分：前70% reference，后30% current
    df_sorted = df.sort_values('created_at')
    split = int(len(df_sorted) * 0.7)
    reference = df_sorted.iloc[:split].drop(columns=['created_at'])
    current   = df_sorted.iloc[split:].drop(columns=['created_at'])

    print(f"Reference: {len(reference):,} rows | Current: {len(current):,} rows")

    # 定义数据schema
    definition = DataDefinition(
        numerical_columns=[
            'created_hour', 'created_dow', 'created_month',
            'hist_avg_resolution_hours'
        ],
        categorical_columns=[
            'agency', 'complaint_type', 'borough',
            'channel_type', 'is_weekend', 'is_overdue'
        ]
    )

    ref_dataset = Dataset.from_pandas(reference, data_definition=definition)
    cur_dataset = Dataset.from_pandas(current,   data_definition=definition)

    # 生成报告
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=ref_dataset, current_data=cur_dataset)
    result.save_html("ml_pipeline/reports/drift_report.html")
    print("Drift report saved → ml_pipeline/reports/drift_report.html")

if __name__ == "__main__":
    run_drift_report()