# %%
# ruff: noqa E402
import pandas as pd
import numpy as np
import os
import sys
import mlflow
import mlflow.sklearn
import warnings
import joblib

from prophet import Prophet
from prophet.serialize import model_to_json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import matplotlib.pyplot as plt

# %%
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# %%
from src.services.predict_all_ticketsv2.feature_engineering import (
    load_and_prepare,
    create_time_features,  # Mantido para salvar os dados para a API
)

# %%
# mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
# mlflow.set_experiment("all_tickets_v3")
# mlflow.autolog(disable=True)


# %%
def plot_predictions(train_data, test_data, predictions, model_name):
    """
    Plota os dados de treino, os valores reais de teste e os valores previstos.

    Args:
        train_data (pd.Series): Os dados de treinamento (valores reais).
        test_data (pd.Series): Os valores reais do conjunto de teste.
        predictions (pd.Series): Os valores previstos pelo modelo para o período de teste.
        model_name (str): O nome do modelo para o título do gráfico.
    """
    plt.figure(figsize=(15, 7))

    # Plotar dados de treino
    plt.plot(
        train_data.index,
        train_data,
        label="Dados de Treino (70%)",
        color="blue",
        linestyle="-",
    )

    # Plotar dados de teste (reais)
    plt.plot(
        test_data.index,
        test_data,
        label="Valores Reais de Teste (30%)",
        color="green",
        marker=".",
        linestyle="-",
    )

    # Plotar previsões
    plt.plot(
        test_data.index,
        predictions,
        label="Valores Previstos",
        color="red",
        marker=".",
        linestyle="--",
    )

    plt.title(f"Histórico, Teste e Previsão - Modelo {model_name}", fontsize=16)
    plt.xlabel("Data", fontsize=12)
    plt.ylabel("Contagem de Tickets", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()


# %%
def train_prophet(df, test_size=0.2):
    """Treina o modelo Prophet e salva o artefato."""
    # with mlflow.start_run(run_name="Prophet"):
    # Preparar os dados para o Prophet (requer colunas 'ds' e 'y')
    df_prophet = df.rename(columns={"date": "ds", "ticket_count": "y"})

    # Dividir em treino e teste
    split_idx = int(len(df_prophet) * (1 - test_size))
    train, test = df_prophet.iloc[:split_idx], df_prophet.iloc[split_idx:]

    # Instanciar e treinar o modelo
    model = Prophet(
        changepoint_prior_scale=0.5,  # Aumenta a flexibilidade da tendência
        seasonality_prior_scale=10.0,  # Padrão, mas pode ser ajustado
        holidays_prior_scale=10.0,  # Aumenta o efeito dos feriados
        seasonality_mode="multiplicative",  # Tenta um modo de sazonalidade diferente
    )

    # Adicionar feriados dos EUA (ajuda a modelar quedas em feriados)
    model.add_country_holidays(country_name="US")

    # Adicionar sazonalidade mensal
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    model.fit(train)

    # Fazer previsões no conjunto de teste
    future = test[["ds"]]
    y_pred = model.predict(future)["yhat"]
    y_test = test["y"]

    # Calcular métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # mlflow.log_metrics({"mae": mae, "mse": mse, "rmse": np.sqrt(mse), "r2": r2})
    # mlflow.log_params(
    #     {
    #         "changepoint_prior_scale": 0.5,
    #         "seasonality_prior_scale": 10.0,
    #         "holidays_prior_scale": 15.0,
    #         "seasonality_mode": "multiplicative",
    #     }
    # )

    # Plotar os resultados
    plot_predictions(train["y"], y_test, y_pred, "Prophet Otimizado")

    # Salvar modelo localmente (modelos Prophet são salvos como JSON)
    models_dir = REPO_ROOT / "models" / "all_tickets_kaggle"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "prophet_model.json"
    with open(model_path, "w") as fout:
        fout.write(model_to_json(model))

    print(f"✓ Modelo Prophet salvo em {model_path}")
    print(f"Prophet - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.4f}")

    return model, y_pred


# %%
def save_processed_data(data):
    """Salva os dados processados que a API irá consumir."""
    processed_dir = REPO_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / "tickets_with_features.csv"
    data.to_csv(output_path, index=False)

    print(f"✓ Dados processados salvos em {output_path}")


# %%
# ==================== FLUXO PRINCIPAL ====================
print("Carregando e preparando os dados...")
df_raw = pd.read_csv(REPO_ROOT / "data" / "rows.csv")
df = load_and_prepare(df_raw)

print("\n" + "=" * 50)
print("Treinando modelo Prophet...")
print("=" * 50 + "\n")

prophet_model, _ = train_prophet(df)

print("\n" + "=" * 50)
print("Salvando dados para a API...")
# A API ainda espera este arquivo para carregar o histórico
df_with_features = create_time_features(df)
save_processed_data(df_with_features)

print("\n" + "=" * 50)
print("Treinamento concluído com sucesso!")
print("=" * 50)
