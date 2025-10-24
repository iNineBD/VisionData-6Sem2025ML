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


# ==================== CONFIGURAÇÕES ====================
MODEL_PATH = "models/all_tickets_kaggle/lightgbm_model.pkl"
SCALER_PATH = "models/all_tickets_kaggle/scaler.pkl"
DATA_PATH = "data/processed/tickets_with_features.csv"

# Variáveis globais
loaded_model = None
loaded_scaler = None
loaded_data = None


# ==================== FUNÇÕES DE CARREGAMENTO ====================
def load_model_and_scaler():
    """Carrega modelo e scaler salvos"""
    global loaded_model, loaded_scaler

    try:
        loaded_model = joblib.load(MODEL_PATH)
        loaded_scaler = joblib.load(SCALER_PATH)
        print("✓ Modelo e scaler carregados com sucesso")
    except Exception as e:
        print(f"✗ Erro ao carregar modelo: {e}")
        raise


def load_historical_data():
    """Carrega dados históricos processados"""
    global loaded_data

    try:
        df = pd.read_csv(DATA_PATH)
        df["date"] = pd.to_datetime(df["date"])
        loaded_data = df.sort_values("date")
        print(f"✓ Dados históricos carregados: {len(df)} registros")
    except Exception as e:
        print(f"✗ Erro ao carregar dados: {e}")
        raise


# ==================== FUNÇÕES DE PREVISÃO ====================
def predict_future(model, scaler, historical_data, days=30):
    """
    Gera previsões para os próximos N dias
    Usa funções compartilhadas - SEM duplicação de código!
    """
    # Preparar dados históricos (apenas date e ticket_count)
    hist_df = historical_data[["date", "ticket_count"]].copy()

    predictions = []
    last_date = hist_df["date"].max()

    for i in range(1, days + 1):
        # Data futura
        future_date = last_date + timedelta(days=i)

        # Criar features usando função compartilhada
        feature_dict = create_single_day_features(future_date, hist_df)

        # Preparar para predição (ordem garantida automaticamente!)
        X_future = prepare_features_for_prediction(feature_dict)

        # Padronizar
        X_future_scaled = scaler.transform(X_future)

        # Prever
        pred = model.predict(X_future_scaled)[0]
        pred = max(0, int(round(pred)))

        predictions.append({"date": future_date, "ticket_count": pred})

        # Adicionar predição ao histórico para próxima iteração
        hist_df = pd.concat(
            [hist_df, pd.DataFrame([{"date": future_date, "ticket_count": pred}])],
            ignore_index=True,
        )

    return pd.DataFrame(predictions)


# ==================== LIFECYCLE ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação FastAPI"""
    # Startup
    print("=" * 60)
    print("🚀 Iniciando API de Previsão de Tickets...")
    print("=" * 60)
    load_model_and_scaler()
    load_historical_data()

    feature_cols = get_feature_columns()
    print(f"📊 Features disponíveis: {len(feature_cols)}")
    print(f"   Primeiras 5: {feature_cols[:5]}")
    print("=" * 60)
    print("✅ API pronta para uso!")
    print("=" * 60)
    yield
    # Shutdown
    print("\n👋 Encerrando API...")


# ==================== FASTAPI APP ====================
app = FastAPI(
    title="API de Previsão de Tickets",
    description="API para previsão de quantidade de tickets usando features compartilhadas",
    version="2.0.0",
    lifespan=lifespan,
)


# ==================== ENDPOINTS ====================
@app.get("/")
async def root():
    """Endpoint raiz com informações da API"""
    return {
        "message": "API de Previsão de Tickets",
        "version": "2.0.0",
        "status": "online",
        "features": "Usando funções compartilhadas (DRY principle)",
        "endpoints": {
            "/predictAllTickets": "GET - Retorna dados históricos e previsões de todos os tickets",
            "/health": "GET - Verifica status da API",
            "/model-info": "GET - Informações sobre o modelo",
        },
    }


@app.get("/health")
async def health_check():
    """Verifica se o modelo e dados estão carregados"""
    feature_cols = get_feature_columns()

    return {
        "status": (
            "healthy"
            if all([loaded_model, loaded_scaler, loaded_data is not None])
            else "unhealthy"
        ),
        "model_loaded": loaded_model is not None,
        "scaler_loaded": loaded_scaler is not None,
        "data_loaded": loaded_data is not None,
        "n_features": len(feature_cols),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/model-info")
async def model_info():
    """Retorna informações sobre o modelo carregado"""
    if loaded_model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado")

    feature_cols = get_feature_columns()

    return {
        "model_type": type(loaded_model).__name__,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "data_records": len(loaded_data) if loaded_data is not None else 0,
        "date_range": {
            "start": (
                loaded_data["date"].min().strftime("%Y-%m-%d")
                if loaded_data is not None
                else None
            ),
            "end": (
                loaded_data["date"].max().strftime("%Y-%m-%d")
                if loaded_data is not None
                else None
            ),
        },
    }


@app.get("/predictAllTickets", response_model=ForecastResponse)
async def get_forecast(days: int = 30, historical_days: int = 90):
    """
    Retorna dados históricos e previsões futuras

    Parameters:
    - days: número de dias para prever (padrão: 30)
    - historical_days: número de dias históricos para retornar (padrão: 90)
    """
    # Validações
    if loaded_model is None or loaded_scaler is None or loaded_data is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado")

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

        # Gerar previsões (usando função simplificada!)
        predictions_df = predict_future(
            loaded_model, loaded_scaler, loaded_data, days=days
        )

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
            historical_data=historical_data,
            predictions=predictions,
            model_used="LightGBM",
            forecast_period_days=days,
            metadata={
                "historical_period_days": historical_days,
                "last_historical_date": historical_df["date"]
                .max()
                .strftime("%Y-%m-%d"),
                "first_prediction_date": predictions_df["date"]
                .min()
                .strftime("%Y-%m-%d"),
                "last_prediction_date": predictions_df["date"]
                .max()
                .strftime("%Y-%m-%d"),
                "total_historical_tickets": total_historical,
                "avg_historical_tickets": round(avg_historical, 2),
                "total_predicted_tickets": total_predicted,
                "avg_predicted_tickets": round(avg_predicted, 2),
                "generated_at": datetime.now().isoformat(),
            },
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
        raise HTTPException(status_code=500, detail="Erro ao gerar previsões (ver logs do servidor).")
    return JSONResponse(status_code=200, content=res)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
