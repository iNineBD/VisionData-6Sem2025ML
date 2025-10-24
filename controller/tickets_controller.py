# ruff: noqa E402
import sys
import os
import pandas as pd
import uvicorn
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from prophet import Prophet
from prophet.serialize import model_from_json
from fastapi.responses import JSONResponse
from src.services.predict_company.app import run_pipeline
from src.config import config

# Adicionar caminho do projeto para importar módulos compartilhados
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Importar funções compartilhadas
from src.services.predict_all_ticketsv2.feature_engineering import (
    create_single_day_features,
    prepare_features_for_prediction,
    get_feature_columns,
)


# ==================== RESPONSE MODELS ====================
class PredictionPoint(BaseModel):
    """Representa um ponto de previsão ou dado histórico"""

    date: str
    ticket_count: int
    is_prediction: bool


class ForecastResponse(BaseModel):
    """Representa a resposta da previsão"""

    historical_data: List[PredictionPoint]
    predictions: List[PredictionPoint]
    model_used: str
    forecast_period_days: int
    metadata: Dict[str, Any]


class PredictionResponse(BaseModel):
    model_name: str
    days: int
    predictions: dict


# ==================== CONFIGURAÇÕES ====================
ACTIVE_MODEL = os.getenv("ACTIVE_MODEL", "prophet").lower()

MODEL_PATHS = {
    "prophet": "models/all_tickets_kaggle/prophet_model.json",
}
DATA_PATH = "data/processed/tickets_with_features.csv"

# Variáveis globais
loaded_model = None
model_type = ACTIVE_MODEL


# ==================== FUNÇÕES DE CARREGAMENTO ====================
def load_resources():
    """Carrega modelo, scaler e dados históricos com base na configuração."""
    global loaded_model, loaded_scaler, loaded_data, model_type

    model_path = MODEL_PATHS.get(ACTIVE_MODEL)
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Arquivo do modelo para '{ACTIVE_MODEL}' não encontrado em '{model_path}'"
        )

    print(f"🚀 Carregando modelo '{ACTIVE_MODEL}' de '{model_path}'...")

    try:
        if ACTIVE_MODEL == "prophet":
            with open(model_path, "r") as fin:
                loaded_model = model_from_json(fin.read())
            print("✓ Modelo Prophet carregado.")
        else:
            raise ValueError(f"Tipo de modelo desconhecido: {ACTIVE_MODEL}")

        # Carregar dados históricos (comum a todos)
        df = pd.read_csv(DATA_PATH)
        df["date"] = pd.to_datetime(df["date"])
        loaded_data = df.sort_values("date")
        print(f"✓ Dados históricos carregados: {len(df)} registros")

    except Exception as e:
        print(f"✗ Erro fatal durante o carregamento: {e}")
        raise


# ==================== FUNÇÕES DE PREVISÃO ====================
def predict_future(days=30):
    """Gera previsões para os próximos N dias usando o modelo carregado."""
    if ACTIVE_MODEL == "prophet":
        return predict_future_prophet(days)
    else:
        raise NotImplementedError(
            f"Lógica de previsão não implementada para {ACTIVE_MODEL}"
        )


def predict_future_prophet(days=30):
    """Gera previsões usando o modelo Prophet."""
    # CORREÇÃO: Criar um dataframe futuro que começa após a última data histórica
    last_date = loaded_data["date"].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    future_df = pd.DataFrame({"ds": future_dates})

    forecast = loaded_model.predict(future_df)

    # Retorna apenas as previsões futuras
    predictions = forecast[["ds", "yhat"]]
    predictions = predictions.rename(columns={"ds": "date", "yhat": "ticket_count"})
    predictions["ticket_count"] = predictions["ticket_count"].apply(
        lambda x: max(0, int(round(x)))
    )
    return predictions


# ==================== LIFECYCLE ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação FastAPI"""
    print("=" * 60)
    print("🚀 Iniciando API de Previsão de Tickets...")
    print("=" * 60)
    load_resources()
    print("=" * 60)
    print("✅ API pronta para uso!")
    print("=" * 60)
    yield
    print("\n👋 Encerrando API...")


# ==================== FASTAPI APP ====================
app = FastAPI(
    title="API de Previsão de Tickets",
    description="API para previsão de quantidade de tickets usando features compartilhadas",
    version="2.0.0",
    lifespan=lifespan,
)


# ==================== ENDPOINTS ====================
@app.get("/predictAllTickets", response_model=ForecastResponse)
async def get_forecast(days: int = 30, historical_days: int = 90):
    """
    Retorna dados históricos e previsões futuras

    Parameters:
    - days: número de dias para prever (padrão: 30)
    - historical_days: número de dias históricos para retornar (padrão: 90)
    """
    # Validações
    model_ok = loaded_model is not None
    data_ok = loaded_data is not None
    scaler_ok = True if ACTIVE_MODEL != "lightgbm" else loaded_scaler is not None

    if not all([model_ok, data_ok, scaler_ok]):
        raise HTTPException(
            status_code=500, detail="Recursos não carregados corretamente"
        )

    if days <= 0 or days > 365:
        raise HTTPException(status_code=400, detail="Dias deve estar entre 1 e 365")

    if historical_days <= 0 or historical_days > len(loaded_data):
        raise HTTPException(
            status_code=400,
            detail=f"Historical_days deve estar entre 1 e {len(loaded_data)}",
        )

    try:
        # Pegar dados históricos recentes
        historical_df = loaded_data.tail(historical_days).copy()

        # Gerar previsões (usando função unificada!)
        predictions_df = predict_future(days=days)

        # Formatar dados históricos
        historical_data = [
            PredictionPoint(
                date=row["date"].strftime("%Y-%m-%d"),
                ticket_count=int(row["ticket_count"]),
                is_prediction=False,
            )
            for _, row in historical_df.iterrows()
        ]

        # Formatar previsões
        predictions = [
            PredictionPoint(
                date=row["date"].strftime("%Y-%m-%d"),
                ticket_count=int(row["ticket_count"]),
                is_prediction=True,
            )
            for _, row in predictions_df.iterrows()
        ]

        # Calcular estatísticas
        total_historical = int(historical_df["ticket_count"].sum())
        avg_historical = float(historical_df["ticket_count"].mean())
        total_predicted = int(predictions_df["ticket_count"].sum())
        avg_predicted = float(predictions_df["ticket_count"].mean())

        return ForecastResponse(
            historical_data=historical_data, predictions=predictions
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar previsão: {str(e)}")


@app.get("/predict_company", response_model=PredictionResponse)
def predict_company():
    """Roda o pipeline do notebook e retorna JSON com os forecasts."""
    if not os.path.exists(config.CSV_PATH):
        raise HTTPException(status_code=400, detail=f"CSV not found: {config.CSV_PATH}")
    res = run_pipeline(config.CSV_PATH)
    if res is None:
        raise HTTPException(
            status_code=500, detail="Erro ao gerar previsões (ver logs do servidor)."
        )
    return JSONResponse(status_code=200, content=res)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
