from pathlib import Path
import pandas as pd
import warnings
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
MODELS_DIR = REPO_ROOT / "models" / "predict_product_tickets"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = REPO_ROOT / "data" / "rows.csv"


def train_lightgbm(series: pd.Series, forecast_days=FORECAST_DAYS):
    df_feat = create_lgb_features(series)
    if df_feat.shape[0] < 50:
        return None
    test_size = min(90, int(len(df_feat) * 0.3))
    train = df_feat.iloc[:-test_size]
    X_train = train.drop(columns=["y"])
    y_train = train["y"]
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
    return model


def train_sarimax(series: pd.Series, seasonal_period=SEASONAL_PERIOD):
    if len(series) < 2:
        return None
    try:
        order = (1, 1, 1)
        seasonal_order = (1, 0, 1, seasonal_period)
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        return res
    except Exception:
        return None


def main():
    print(f"Lendo dados de {CSV_PATH} ...")
    df = parse_and_prep(str(CSV_PATH), "Product")
    top_products = df["Product"].value_counts().head(5).index.tolist()
    for prod in top_products:
        print(f"Treinando modelos para: {prod}")
        series = make_daily_series(df, prod, "Product")
        if series.empty or len(series) < 50:
            print(f"- Dados insuficientes para {prod}")
            continue
        sarimax_model = train_sarimax(series)
        lgbm_model = train_lightgbm(series)
        if sarimax_model:
            sarimax_path = MODELS_DIR / f"{prod}_SARIMAX.pkl"
            joblib.dump(sarimax_model, sarimax_path)
            print(f"  ✓ SARIMAX salvo em {sarimax_path}")
        if lgbm_model:
            lgbm_path = MODELS_DIR / f"{prod}_LightGBM.pkl"
            joblib.dump(lgbm_model, lgbm_path)
            print(f"  ✓ LightGBM salvo em {lgbm_path}")


if __name__ == "__main__":
    main()
