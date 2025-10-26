# app.py
import os
import io
import pandas as pd
import numpy as np
from datetime import timedelta
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from src.config import config
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import joblib
from pathlib import Path


# Model libs
from statsmodels.tsa.statespace.sarimax import SARIMAX
import lightgbm as lgb
from src.utils.data_processing import parse_and_prep, make_daily_series, create_lgb_features

warnings.filterwarnings("ignore")

FORECAST_DAYS=30
SEASONAL_PERIOD=7
REPO_ROOT = Path(__file__).resolve().parents[3]

models_dir = REPO_ROOT / "models" / "predict_company_tickets"
models_dir.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Predição Tickets - SARIMAX + LightGBM")


def train_lightgbm(series: pd.Series, forecast_days=FORECAST_DAYS):
    df_feat = create_lgb_features(series)
    if df_feat.shape[0] < 50:
        return None, None
    test_size = min(90, int(len(df_feat) * 0.3))
    train = df_feat.iloc[:-test_size]
    test = df_feat.iloc[-test_size:]
    X_train = train.drop(columns=["y"])
    y_train = train["y"]
    X_test = test.drop(columns=["y"])
    y_test = test["y"]
    lgb_train = lgb.Dataset(X_train, y_train)
    params = {
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 500,
    }
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train],
        callbacks=[lgb.log_evaluation(0)]
    )
    y_pred_test = model.predict(X_test)
    last_known = series.copy()
    future_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="D")
    preds = []
    tmp_series = last_known.copy()
    for dt in future_index:
        feats = {}
        for lag in [1,7,14,30]:
            lag_date = dt - pd.Timedelta(days=lag)
            feats[f"lag_{lag}"] = tmp_series.get(lag_date, 0)
        for w in [7,30]:
            window_vals = [tmp_series.get(dt - pd.Timedelta(days=i), 0) for i in range(1, w+1)]
            feats[f"roll_mean_{w}"] = np.mean(window_vals) if window_vals else 0
            feats[f"roll_std_{w}"] = np.std(window_vals) if window_vals else 0
        feats["dayofweek"] = dt.dayofweek
        feats["day"] = dt.day
        feats["month"] = dt.month
        Xf = pd.DataFrame([feats])
        p = model.predict(Xf)[0]
        p = max(0, p)
        preds.append(p)
        tmp_series[dt] = p
    mse = mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_test)
    return {
        "model": model,
        "preds": pd.Series(preds, index=future_index),
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "feature_importance": model.feature_importance().tolist(),
        "features": list(X_train.columns),
        "y_test": y_test,
        "y_pred_test": pd.Series(y_pred_test, index=y_test.index)
    }

def train_sarimax(series: pd.Series, seasonal_period=SEASONAL_PERIOD, forecast_days=FORECAST_DAYS):
    if len(series) < 2:
        return None
    try:
        test_days = min(90, int(len(series)*0.3))
        train = series.iloc[:-test_days] if test_days>0 else series
        test = series.iloc[-test_days:] if test_days>0 else series
        order = (1,1,1)
        seasonal_order = (1,0,1,seasonal_period)
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        pred_test = pd.Series(dtype=float)
        if test_days>0:
            pred_test = res.get_prediction(start=test.index[0], end=test.index[-1]).predicted_mean
            mse = mean_squared_error(test, pred_test)
            mae = mean_absolute_error(test, pred_test)
            rmse = np.sqrt(mse)
            r2 = r2_score(test, pred_test) if len(test)>1 else float("nan")
        else:
            mse = mae = rmse = r2 = float("nan")
        future_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="D")
        forecast = res.get_forecast(steps=forecast_days).predicted_mean
        forecast.index = future_index
        return {
            "model": res,
            "preds": forecast,
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "aic": float(res.aic) if hasattr(res, "aic") else None,
            "bic": float(res.bic) if hasattr(res, "bic") else None,
            "order": order,
            "seasonal_order": seasonal_order,
            "y_test": test,
            "y_pred_test": pred_test
        }
    except Exception as e:
        return None

def run_pipeline(csv_path: str, group_col: str):
    """Executa o pipeline de previsão para a coluna especificada (empresa ou produto)."""

    models_dir = REPO_ROOT / "models" / f"predict_{group_col.lower()}_tickets"
    models_dir.mkdir(parents=True, exist_ok=True)
    df = parse_and_prep(csv_path, group_col)
    top_values = df[group_col].value_counts().head(5).index.tolist()
    forecasts_summary = {}

    for item in top_values:
        existing_models = [
            f for f in os.listdir(models_dir)
            if f.startswith(item) and f.endswith(".pkl")
        ]
        if existing_models:
            model_file = os.path.join(models_dir, existing_models[0])
            model_name = "SARIMAX" if "SARIMAX" in model_file else "LightGBM"
            series = make_daily_series(df, item, group_col)
            if series.empty or len(series) < 50:
                continue

            forecast_days = FORECAST_DAYS
            future_index = pd.date_range(
                start=series.index[-1] + pd.Timedelta(days=1),
                periods=forecast_days, freq="D"
            )

            model = joblib.load(model_file)
            if model_name == "LightGBM":
                tmp_series = series.copy()
                preds = []
                for dt in future_index:
                    feats = {}
                    for lag in [1, 7, 14, 30]:
                        feats[f"lag_{lag}"] = tmp_series.get(dt - pd.Timedelta(days=lag), 0)
                    for w in [7, 30]:
                        vals = [tmp_series.get(dt - pd.Timedelta(days=i), 0) for i in range(1, w + 1)]
                        feats[f"roll_mean_{w}"] = np.mean(vals)
                        feats[f"roll_std_{w}"] = np.std(vals)
                    feats["dayofweek"] = dt.dayofweek
                    feats["day"] = dt.day
                    feats["month"] = dt.month
                    Xf = pd.DataFrame([feats])
                    p = model.predict(Xf)[0]
                    p = max(0, p)
                    preds.append(p)
                    tmp_series[dt] = p
                preds = pd.Series(preds, index=future_index)
            else:
                preds = model.get_forecast(steps=forecast_days).predicted_mean
                preds.index = future_index

            total_pred = float(preds.sum())
            last_30_sum = float(series.iloc[-30:].sum())
            pct_increase = ((total_pred - last_30_sum) / last_30_sum * 100) if last_30_sum > 0 else None

            forecasts_summary[item] = {
                "best_model": model_name,
                "reason": "Modelo existente reutilizado",
                # "mse": None, "mae": None, "rmse": None, "r2": None,
                "total_next30": total_pred,
                "pct_increase": pct_increase,
                "forecast": preds.to_dict(),
                "raw_series": series.tail(60).to_dict(),
            }
            continue

        series = make_daily_series(df, item, group_col)
        if series.empty or len(series) < 50:
            continue

        sar = train_sarimax(series, seasonal_period=SEASONAL_PERIOD)
        lgbm = train_lightgbm(series)

        def get_score(m):
            if not m:
                return float("inf")
            return (m["mse"] + m["mae"]) / 2 - m["r2"]

        score_sar, score_lgb = get_score(sar), get_score(lgbm)
        best_model, best = ("SARIMAX", sar) if score_sar < score_lgb else ("LightGBM", lgbm)

        if best and best.get("model"):
            model_path = os.path.join(models_dir, f"{item}_{best_model}.pkl")
            try:
                joblib.dump(best["model"], model_path)
            except Exception as e:
                print(f"⚠️ Erro ao salvar modelo: {e}")

        preds = best.get("preds", pd.Series(dtype=float))
        total_pred = float(preds.sum()) if not preds.empty else None
        last_30_sum = float(series.iloc[-30:].sum())

        forecasts_summary[item] = {
            "best_model": best_model,
            "reason": "Treinado novo modelo",
            # "mse": best.get("mse"), "mae": best.get("mae"),
            # "rmse": best.get("rmse"), "r2": best.get("r2"),
            "total_next30": total_pred,
            "pct_increase": ((total_pred - last_30_sum) / last_30_sum * 100) if last_30_sum > 0 else None,
            "forecast": preds.to_dict(),
            "raw_series": series.tail(60).to_dict(),
        }

    def serialize_series_dict(d):
        return {str(k): int(round(vv)) for k, vv in (d or {}).items() if not pd.isna(vv)}

    final_summary = [
        {
            group_col.lower(): item,
            **v,
            "total_next30": int(round(v["total_next30"])) if v.get("total_next30") else None,
            "forecast": serialize_series_dict(v.get("forecast")),
            "raw_series": serialize_series_dict(v.get("raw_series")),
        }
        for item, v in forecasts_summary.items()
    ]
    return {"best_models_summary": final_summary}

def load_and_predict(csv_path=config.CSV_PATH, forecast_days=FORECAST_DAYS, model_dir="models"):
    """Carrega os modelos salvos e gera previsões rápidas sem reentreinar."""

    df = parse_and_prep(csv_path)
    top_companies = df[config.COMPANY_COL].value_counts().head(5).index.tolist()

    results = {}
    for comp in top_companies:
        model_files = [f for f in os.listdir(model_dir) if f.startswith(comp)]
        if not model_files:
            continue

        model_file = os.path.join(model_dir, model_files[0])
        model = joblib.load(model_file)

        series = make_daily_series(df, comp)
        if series.empty:
            continue

        future_index = pd.date_range(
            start=series.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D"
        )

        if "LightGBM" in model_file:
            tmp_series = series.copy()
            preds = []
            for dt in future_index:
                feats = {}
                for lag in [1, 7, 14, 30]:
                    feats[f"lag_{lag}"] = tmp_series.get(dt - pd.Timedelta(days=lag), 0)
                for w in [7, 30]:
                    vals = [tmp_series.get(dt - pd.Timedelta(days=i), 0) for i in range(1, w + 1)]
                    feats[f"roll_mean_{w}"] = np.mean(vals)
                    feats[f"roll_std_{w}"] = np.std(vals)
                feats["dayofweek"] = dt.dayofweek
                feats["day"] = dt.day
                feats["month"] = dt.month

                Xf = pd.DataFrame([feats])
                p = model.predict(Xf)[0]
                p = max(0, p)
                preds.append(p)
                tmp_series[dt] = p
            preds = pd.Series(preds, index=future_index)

        elif "SARIMAX" in model_file:
            preds = model.get_forecast(steps=forecast_days).predicted_mean
            preds.index = future_index

        results[comp] = preds.to_dict()

    return results



# @app.get("/download_metrics")
# def download_metrics():
#     if not os.path.exists(METRICS_CSV):
#         raise HTTPException(status_code=404, detail="metrics CSV não encontrado. Rode /predict_top5 primeiro.")
#     return FileResponse(METRICS_CSV, media_type="text/csv", filename=METRICS_CSV)

