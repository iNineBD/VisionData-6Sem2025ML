# %%
# ruff: noqa E402
import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
import mlflow
import mlflow.sklearn
import warnings
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from mlflow.models.signature import infer_signature
from pathlib import Path

# %%
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

sns.set_theme(style="whitegrid")

# %%
from src.services.predict_all_ticketsv2.feature_engineering import (
    load_and_prepare,
    create_time_features,
)

# %%
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("all_tickets_v3")
mlflow.autolog(disable=True)


# %%
def split_train_test(df, test_size=0.2):
    """Divide os dados em conjuntos de treino e teste, padronizando as features."""

    # ordenar por data
    df = df.sort_values("date")

    # remover linhas com NaN
    df_clean = df.dropna().reset_index(drop=True)

    # definindo X e y somente com features importantes
    X = df_clean.drop(columns=["ticket_count", "date"])
    y = df_clean["ticket_count"]

    # Dividir em conjuntos de treino e teste
    split_idx = int(len(df_clean) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Padronizar as features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        # Padronizar as features
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        # Padronizar as features
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# %%
def train_lightgbm(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="LightGBM"):
        model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbose=-1,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_params({"n_estimators": 100, "learning_rate": 0.05, "max_depth": 5})
        mlflow.log_metrics({"mae": mae, "mse": mse, "rmse": np.sqrt(mse), "r2": r2})

        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(5)
        mlflow.sklearn.log_model(
            model,
            name="model_lightgbm",
            signature=signature,
            input_example=input_example,
        )

        # salvar modelo localmente (usar path absoluto)
        models_dir = REPO_ROOT / "models" / "all_tickets_kaggle"
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, models_dir / "lightgbm_model.pkl")

        print(f"LightGBM - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.4f}")

        return model, y_pred


# %%
def train_sarimax(df, test_size=0.2):
    with mlflow.start_run(run_name="SARIMAX"):
        # Preparar os dados
        df_sorted = df.sort_values("date")
        ts = df_sorted.set_index("date")["ticket_count"]

        # Dividir em treino e teste
        split_idx = int(len(ts) * (1 - test_size))
        train, test = ts.iloc[:split_idx], ts.iloc[split_idx:]

        model = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        fitted = model.fit(disp=False)

        y_pred = fitted.forecast(steps=len(test))

        mae = mean_absolute_error(test, y_pred)
        mse = mean_squared_error(test, y_pred)
        r2 = r2_score(test, y_pred)

        mlflow.log_params({"order": "(1,1,1)", "seasonal_order": "(1,1,1,7)"})
        mlflow.log_metrics({"mae": mae, "mse": mse, "rmse": np.sqrt(mse), "r2": r2})

        mlflow.statsmodels.log_model(fitted, name="model_sarimax")

        # salvar modelo localmente
        models_dir = REPO_ROOT / "models" / "all_tickets_kaggle"
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted, models_dir / "sarimax_model.pkl")

        print(f"SARIMAX - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.4f}")

        return fitted, y_pred


# %%
def train_all_models(df):
    print("Preparando dados...")

    df_features = create_time_features(df)

    X_train, X_test, y_train, y_test, scaler = split_train_test(df_features)

    print("\n" + "=" * 50)
    print("Treinando modelos...")
    print("=" * 50 + "\n")

    results = {}

    print("1. LightGBM")
    lgbm_model, lgbm_pred = train_lightgbm(X_train, X_test, y_train, y_test)
    results["lightgbm"] = {
        "model": lgbm_model,
        "predictions": lgbm_pred,
        "y_test": y_test,
    }

    print("\n2. SARIMAX")
    sarimax_model, sarimax_pred = train_sarimax(df)
    results["sarimax"] = {"model": sarimax_model, "predictions": sarimax_pred}

    print("\n" + "=" * 50)
    print("Treinamento concluído!")
    print("=" * 50)

    return results, scaler


# %%
df = load_and_prepare("../../../data/rows.csv")

results, scaler = train_all_models(df)

lgbm_model = results["lightgbm"]["model"]
sarimax_model = results["sarimax"]["model"]


# %%
def save_for_api(scaler, data, output_dir="../../../models/all_tickets_kaggle/"):
    """
    Salva apenas o necessário para a API
    NÃO precisa mais salvar feature_order!
    """
    from pathlib import Path

    if output_dir is None:
        output_dir = REPO_ROOT / "models" / "all_tickets_kaggle"
    else:
        output_dir = Path(output_dir)

    processed_dir = REPO_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Salvar scaler (NECESSÁRIO)
    joblib.dump(scaler, output_dir / "scaler.pkl")

    # Salvar dados processados (NECESSÁRIO)
    data.to_csv(processed_dir / "tickets_with_features.csv", index=False)

    print(f"✓ Scaler salvo em {output_dir / 'scaler.pkl'}")
    print(f"✓ Dados salvos em {processed_dir / 'tickets_with_features.csv'}")


# %%
save_for_api(
    scaler,
    create_time_features(df),
    output_dir=REPO_ROOT / "models" / "all_tickets_kaggle",
)
