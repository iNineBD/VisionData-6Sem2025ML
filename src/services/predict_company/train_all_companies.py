from pathlib import Path
import pandas as pd
import warnings
import mlflow
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.data_processing import (
    parse_and_prep,
    make_daily_series,
    create_lgb_features,
)
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
import lightgbm as lgb

warnings.filterwarnings("ignore")

FORECAST_DAYS = 30
SEASONAL_PERIOD = 7
REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "models" / "predict_company_tickets"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = REPO_ROOT / "data" / "rows.csv"


def train_lightgbm(series: pd.Series, forecast_days=FORECAST_DAYS):
    df_feat = create_lgb_features(series)
    if df_feat.shape[0] < 50:
        return None
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
        params, lgb_train, valid_sets=[lgb_train], callbacks=[lgb.log_evaluation(0)]
    )
    y_pred_test = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = mean_squared_error(y_test, y_pred_test) ** 0.5
    r2 = r2_score(y_test, y_pred_test)
    return {"model": model, "mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def train_sarimax(series: pd.Series, seasonal_period=SEASONAL_PERIOD):
    if len(series) < 2:
        return None
    try:
        test_days = min(90, int(len(series) * 0.3))
        train = series.iloc[:-test_days] if test_days > 0 else series
        test = series.iloc[-test_days:] if test_days > 0 else series
        order = (1, 1, 1)
        seasonal_order = (1, 0, 1, seasonal_period)
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        if test_days > 0:
            pred_test = res.get_prediction(
                start=test.index[0], end=test.index[-1]
            ).predicted_mean
            mse = mean_squared_error(test, pred_test)
            mae = mean_absolute_error(test, pred_test)
            rmse = mean_squared_error(test, pred_test) ** 0.5
            r2 = r2_score(test, pred_test)
        else:
            mse = mae = rmse = r2 = float("nan")
        return {"model": res, "mse": mse, "mae": mae, "rmse": rmse, "r2": r2}
    except Exception:
        return None


def main():
    # Configurar MLflow para apontar para o servidor do docker-compose
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("predict_company_tickets")
    print(f"Lendo dados de {CSV_PATH} ...")
    df = parse_and_prep(str(CSV_PATH), "Company")
    top_companies = df["Company"].value_counts().head(5).index.tolist()
    for comp in top_companies:
        print(f"Treinando modelos para: {comp}")
        series = make_daily_series(df, comp, "Company")
        if series.empty or len(series) < 50:
            print(f"- Dados insuficientes para {comp}")
            continue
        sarimax = train_sarimax(series)
        lgbm = train_lightgbm(series)
        best = None
        with mlflow.start_run(run_name=comp):
            if sarimax:
                mlflow.log_metrics(
                    {
                        "sarimax_mse": sarimax["mse"],
                        "sarimax_mae": sarimax["mae"],
                        "sarimax_rmse": sarimax["rmse"],
                        "sarimax_r2": sarimax["r2"],
                    }
                )
            if lgbm:
                mlflow.log_metrics(
                    {
                        "lgbm_mse": lgbm["mse"],
                        "lgbm_mae": lgbm["mae"],
                        "lgbm_rmse": lgbm["rmse"],
                        "lgbm_r2": lgbm["r2"],
                    }
                )
            if sarimax and lgbm:
                if sarimax["mse"] <= lgbm["mse"]:
                    best = sarimax["model"]
                    print(
                        f"  ✓ SARIMAX é o melhor para {comp} (MSE={sarimax['mse']:.2f} < {lgbm['mse']:.2f})"
                    )
                    mlflow.sklearn.log_model(
                        sarimax["model"],
                        "model",
                        registered_model_name=f"{comp}_SARIMAX",
                    )
                else:
                    best = lgbm["model"]
                    print(
                        f"  ✓ LightGBM é o melhor para {comp} (MSE={lgbm['mse']:.2f} < {sarimax['mse']:.2f})"
                    )
                    mlflow.sklearn.log_model(
                        lgbm["model"], "model", registered_model_name=f"{comp}_LGBM"
                    )
            elif sarimax:
                best = sarimax["model"]
                print(f"  ✓ Apenas SARIMAX treinado para {comp}")
                mlflow.sklearn.log_model(
                    sarimax["model"], "model", registered_model_name=f"{comp}_SARIMAX"
                )
            elif lgbm:
                best = lgbm["model"]
                print(f"  ✓ Apenas LightGBM treinado para {comp}")
                mlflow.sklearn.log_model(
                    lgbm["model"], "model", registered_model_name=f"{comp}_LGBM"
                )
            else:
                print(f"  ✗ Nenhum modelo treinado para {comp}")
                continue
        # Salvar apenas o melhor localmente
        best_path = MODELS_DIR / f"{comp}_BEST.pkl"
        joblib.dump(best, best_path)
        print(f"  ✓ Modelo campeão salvo em {best_path}")


if __name__ == "__main__":
    main()
