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

# Model libs
from statsmodels.tsa.statespace.sarimax import SARIMAX
import lightgbm as lgb
from src.utils.data_processing import parse_and_prep, make_daily_series, create_lgb_features

warnings.filterwarnings("ignore")

FORECAST_DAYS=30
SEASONAL_PERIOD=7

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

def run_pipeline(csv_path=config.CSV_PATH):
    df = parse_and_prep(csv_path)
    top_companies = (
        df[config.COMPANY_COL]
        .value_counts()
        .head(5)
        .index
        .tolist()
    )

    # results = []
    metrics_rows = []
    forecasts_summary = {}
    for comp in top_companies:
        series = make_daily_series(df, comp)
        if series.empty or len(series) < 50: 
            continue
        sar = train_sarimax(series, seasonal_period=SEASONAL_PERIOD)
        lgbm = train_lightgbm(series)

        # Predições e totais previstos
        preds_sar = sar["preds"] if sar else pd.Series(dtype=float)
        preds_lgb = lgbm["preds"] if lgbm else pd.Series(dtype=float)

        total_sar = float(preds_sar.sum()) if not preds_sar.empty else None
        total_lgb = float(preds_lgb.sum()) if not preds_lgb.empty else None

        # Últimos 30 dias
        last_30_start = series.index[-1] - pd.Timedelta(days=29)
        last_30_sum = float(series.loc[last_30_start:series.index[-1]].sum())

        inc_sar_pct = (
            ((total_sar - last_30_sum) / last_30_sum * 100)
            if (total_sar is not None and last_30_sum > 0)
            else None
        )
        inc_lgb_pct = (
            ((total_lgb - last_30_sum) / last_30_sum * 100)
            if (total_lgb is not None and last_30_sum > 0)
            else None
        )

        def get_score(m):
            if not m:
                return float("inf")  # penaliza modelo inexistente
            return (m["mse"] + m["mae"]) / 2 - m["r2"]  # combina erro e r2 (quanto menor, melhor)

        score_sar = get_score(sar)
        score_lgb = get_score(lgbm)
        if score_sar < score_lgb:
            best_model = "SARIMAX"
            best = sar
            reason = "SARIMAX escolhido por menor MSE/MAE e maior R²"
        else:
            best_model = "LightGBM"
            best = lgbm
            reason = "LightGBM escolhido por menor MSE/MAE e maior R²"
        best_info = {
            "best_model": best_model,
            "reason": reason,
            "mse": best["mse"] if best else None,
            "mae": best["mae"] if best else None,
            "rmse": best["rmse"] if best else None,
            "r2": best["r2"] if best else None,
            "total_next30": total_sar if best_model == "SARIMAX" else total_lgb,
            "pct_increase": inc_sar_pct if best_model == "SARIMAX" else inc_lgb_pct,
            "forecast": (preds_sar if best_model == "SARIMAX" else preds_lgb).to_dict(),
            "raw_series": series.to_dict(),
            "y_test": (sar["y_test"] if sar else pd.Series(dtype=float)).to_dict() if sar else {},
            "y_pred_test": (sar["y_pred_test"] if sar else pd.Series(dtype=float)).to_dict() if sar else {},
        }
        forecasts_summary[comp] = best_info
        metrics_rows.append({
            "company": comp,
            "last_30_sum": last_30_sum,
            "best_model": best_model,
            **{f"sar_{k}": sar.get(k) if sar else None for k in ["mse", "mae", "rmse", "r2"]},
            **{f"lgb_{k}": lgbm.get(k) if lgbm else None for k in ["mse", "mae", "rmse", "r2"]}
        })

    # Salva métricas CSV
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(config.METRICS_CSV, index=False)
    final_summary = []
    for comp, v in forecasts_summary.items():
        def serialize_series_dict(d):
            return {
                str(k): float(vv) 
                for k, vv in (d or {}).items() 
                if not pd.isna(vv)
            }
        
        final_summary.append({
            "company": comp,
            "best_model": v.get("best_model"),
            "mse": v.get("mse"),
            "mae": v.get("mae"),
            "rmse": v.get("rmse"),
            "r2": v.get("r2"),
            "total_next30": v.get("total_next30"),
            "pct_increase": v.get("pct_increase"),
            "forecast": serialize_series_dict(v.get("forecast")),
            "raw_series": serialize_series_dict(v.get("raw_series")),
            "y_test": serialize_series_dict(v.get("y_test")),
            "y_pred_test": serialize_series_dict(v.get("y_pred_test")),
        })
    res = {
        "best_models_summary": final_summary
    }

    return res
# @app.get("/download_metrics")
# def download_metrics():
#     if not os.path.exists(METRICS_CSV):
#         raise HTTPException(status_code=404, detail="metrics CSV não encontrado. Rode /predict_top5 primeiro.")
#     return FileResponse(METRICS_CSV, media_type="text/csv", filename=METRICS_CSV)

