import duckdb
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, roc_auc_score
)
import xgboost as xgb
import optuna
import warnings
warnings.filterwarnings('ignore')

# ── 1. 读取数据 ──────────────────────────────────────────
DB_PATH = "data/processed/nyc311.duckdb"

def load_data():
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("""
        SELECT
            resolution_hours,
            is_overdue,
            agency,
            complaint_type,
            borough,
            channel_type,
            created_hour,
            created_dow,
            created_month,
            is_weekend,
            hist_avg_resolution_hours
        FROM mart_ml_features
        WHERE resolution_hours IS NOT NULL
    """).df()
    con.close()
    print(f"Loaded {len(df):,} rows")
    return df

# ── 2. 特征工程 ──────────────────────────────────────────
def prepare_features(df):
    cat_cols = ['agency', 'complaint_type', 'borough', 'channel_type']
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = [
        'agency', 'complaint_type', 'borough', 'channel_type',
        'created_hour', 'created_dow', 'created_month',
        'is_weekend', 'hist_avg_resolution_hours'
    ]

    return df, feature_cols, encoders

# ── 3. Optuna超参调优 ────────────────────────────────────
def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
        'max_depth':        trial.suggest_int('max_depth', 3, 8),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)

# ── 4. 主训练流程 ────────────────────────────────────────
def train():
    mlflow.set_experiment("nyc311-overdue-prediction")

    df = load_data()
    df, feature_cols, encoders = prepare_features(df)

    X = df[feature_cols]
    y = df['is_overdue']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"Overdue rate: {y.mean():.1%}")

    # Optuna调优（20次trials）
    print("\nRunning hyperparameter tuning (20 trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=20,
        show_progress_bar=True
    )

    best_params = study.best_params
    print(f"\nBest AUC: {study.best_value:.4f}")
    print(f"Best params: {best_params}")

    # 用最优参数训练最终模型
    with mlflow.start_run():
        best_params['random_state'] = 42
        best_params['use_label_encoder'] = False
        best_params['eval_metric'] = 'logloss'

        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train, verbose=False)

        # 评估
        test_preds = model.predict(X_test)
        test_proba = model.predict_proba(X_test)[:, 1]
        auc   = roc_auc_score(y_test, test_proba)
        report = classification_report(y_test, test_preds, output_dict=True)

        # Log到MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("test_auc",       auc)
        mlflow.log_metric("test_precision", report['1']['precision'])
        mlflow.log_metric("test_recall",    report['1']['recall'])
        mlflow.log_metric("test_f1",        report['1']['f1-score'])
        mlflow.sklearn.log_model(model, "model")

        print(f"\n── Test Results ──")
        print(f"AUC:       {auc:.4f}")
        print(f"Precision: {report['1']['precision']:.4f}")
        print(f"Recall:    {report['1']['recall']:.4f}")
        print(f"F1:        {report['1']['f1-score']:.4f}")

        # 保存测试集供Evidently用
        X_test_save = X_test.copy()
        X_test_save['is_overdue'] = y_test.values
        X_test_save.to_parquet("ml_pipeline/data/test_reference.parquet", index=False)

        print("\nDone! Run `mlflow ui` to see results.")

if __name__ == "__main__":
    train()
