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
from fastapi.middleware.cors import CORSMiddleware
import logging
from fastapi.responses import FileResponse
from src.utils.dash_export import  generate_forecast_pdf
from src.utils.metrics_go import extract_metric, prepare_chart_data, plot_pie, plot_bar, get_tickets, token, BASE_URL, plot_line_qtd_month, qtd_month, qtd_tkt_priority, plot_line_qtd_priority_month, qtd_tkt_status, plot_line_qtd_status_month
from src.services.predict_company.analyse_company import plot_results
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# Adicionar caminho do projeto para importar módulos compartilhados
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


# Importar funções compartilhadas
from src.utils.feature_engineering import (
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

# ==================== CORS ====================
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


# Utilitários para carregar modelos e prever rapidamente
def load_best_model_and_predict(
    group_col: str,
    models_dir: str,
    csv_path: str,
    forecast_days: int = 30,
    historical_days: int = 60,
):
    from src.utils.data_processing import (
        parse_and_prep,
        make_daily_series,
        _format_series_dict,
    )
    import numpy as np
    df = parse_and_prep(csv_path, group_col)
    top_values = df[group_col].value_counts().head(5).index.tolist()
    results = []
    for item in top_values:
        # Procura apenas o modelo campeão
        best_path = os.path.join(models_dir, f"{item}_BEST.pkl")
        series = make_daily_series(df, item, group_col)
        if series.empty or len(series) < 50 or not os.path.exists(best_path):
            continue
        model = joblib.load(best_path)
        # Detecta tipo do modelo
        model_name = type(model).__name__
        future_index = pd.date_range(
            start=series.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D",
        )
        if hasattr(model, "get_forecast"):
            preds = model.get_forecast(steps=forecast_days).predicted_mean
            preds.index = future_index
        else:
            tmp_series = series.copy()
            preds_list = []
            for dt in future_index:
                feats = {}
                for lag in [1, 7, 14, 30]:
                    feats[f"lag_{lag}"] = tmp_series.get(dt - pd.Timedelta(days=lag), 0)
                for w in [7, 30]:
                    vals = [
                        tmp_series.get(dt - pd.Timedelta(days=i), 0)
                        for i in range(1, w + 1)
                    ]
                    feats[f"roll_mean_{w}"] = np.mean(vals)
                    feats[f"roll_std_{w}"] = np.std(vals)
                feats["dayofweek"] = dt.dayofweek
                feats["day"] = dt.day
                feats["month"] = dt.month
                Xf = pd.DataFrame([feats])
                p = model.predict(Xf)[0]
                p = max(0, p)
                preds_list.append(p)
                tmp_series[dt] = p
            preds = pd.Series(preds_list, index=future_index)
        hist_series = series.tail(historical_days)
        hist_dict = _format_series_dict(hist_series.to_dict())
        pred_dict = _format_series_dict(preds.to_dict()) if preds is not None else {}
        results.append(
            {
                group_col.lower(): item,
                "model_name": model_name,
                "days": forecast_days,
                "historical": hist_dict,
                "predictions": pred_dict,
            }
        )
    return {"best_models_summary": results}


# Endpoint para companhias
@app.get("/predict_company", response_model=PredictionResponse)
def predict_company(days: int = 30, historical_days: int = 60):
    models_dir = "models/predict_company_tickets"
    csv_path = "data/rows.csv"
    if not os.path.exists(models_dir):
        raise HTTPException(
            status_code=500,
            detail="Modelos de companhia não encontrados. Rode o treinamento primeiro.",
        )
    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=400, detail=f"Arquivo CSV não encontrado: {csv_path}"
        )
    res = load_best_model_and_predict(
        "Company",
        models_dir,
        csv_path,
        forecast_days=days,
        historical_days=historical_days,
    )
    if not res["best_models_summary"]:
        raise HTTPException(
            status_code=500,
            detail="Nenhuma previsão gerada. Verifique os dados ou os modelos.",
        )
    return JSONResponse(status_code=200, content=res)


# Endpoint para produtos
@app.get("/predict_product", response_model=PredictionResponse)
def predict_product(days: int = 30, historical_days: int = 60):
    models_dir = "models/predict_product_tickets"
    csv_path = "data/rows.csv"
    if not os.path.exists(models_dir):
        raise HTTPException(
            status_code=500,
            detail="Modelos de produto não encontrados. Rode o treinamento primeiro.",
        )
    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=400, detail=f"Arquivo CSV não encontrado: {csv_path}"
        )
    res = load_best_model_and_predict(
        "Product",
        models_dir,
        csv_path,
        forecast_days=days,
        historical_days=historical_days,
    )
    if not res["best_models_summary"]:
        raise HTTPException(
            status_code=500,
            detail="Nenhuma previsão gerada. Verifique os dados ou os modelos.",
        )
    return JSONResponse(status_code=200, content=res)

#Endpoint para exportar PDF de previsões
@app.get("/export_forecast_pdf", response_model=PredictionResponse)
def export_forecast_pdf(days: int = 30, historical_days: int = 60):
    """
    Gera um PDF único contendo gráficos de:
    - Previsões por Company
    - Previsões por Product
    Reutilizando 100% da lógica já existente.
    """
    res_company = load_best_model_and_predict(
        "Company",
        models_dir="models/predict_company_tickets",
        csv_path="data/rows.csv",
        forecast_days=days,
        historical_days=historical_days,
    )

    res_product = load_best_model_and_predict(
        "Product",
        models_dir="models/predict_product_tickets",
        csv_path="data/rows.csv",
        forecast_days=days,
        historical_days=historical_days,
    )
    if not res_company["best_models_summary"] and not res_product["best_models_summary"]:
        raise HTTPException(status_code=500, detail="Nenhuma previsão disponível.")

    forecasts_summary = {}

    for item in res_company["best_models_summary"]:
        company = item.get("company") or item.get("Company")
        forecasts_summary[f"Company - {company}"] = {
            "raw_series": item["historical"],
            "forecast": item["predictions"],
            "best_model": item["model_name"],
        }

    for item in res_product["best_models_summary"]:
        product = item.get("product") or item.get("Product")
        forecasts_summary[f"Product - {product}"] = {
            "raw_series": item["historical"],
            "forecast": item["predictions"],
            "best_model": item["model_name"],
        }

    saved_charts = plot_results(
        forecasts_summary=forecasts_summary
    )

    pdf_filename = f"relatorio_previsoes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    generate_forecast_pdf(saved_charts, output_file=pdf_filename)

    return FileResponse(
        pdf_filename,
        media_type="application/pdf",
        filename=pdf_filename,
    )

# Endpoint para exportar PDF de métricas
@app.get("/export_metrics_pdf")
def export_metrics_pdf():
    tickets = get_tickets(BASE_URL, token)

    if "data" not in tickets:
        raise HTTPException(500, "Retorno inválido da API de métricas")

    data = tickets["data"]

    pdf_filename = f"relatorio_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Relatório de Métricas de Tickets</b>", styles["Title"]))
    story.append(Spacer(1, 20))

    total = data["totalTickets"]
    story.append(Paragraph(f"<b>Total de Tickets:</b> {total}", styles["Heading2"]))
    story.append(Spacer(1, 12))

    os.makedirs("charts_metrics", exist_ok=True)
    plot_line_qtd_month(qtd_month, "charts_metrics/tickets_by_month.png")

    charts = [
        ("Tickets por Canal", "charts_metrics/tickets_by_channel.png", "TicketsByChannel"),
        ("Tickets por Categoria", "charts_metrics/tickets_by_category.png", "TicketsByCategory"),
        ("Tickets por Tag", "charts_metrics/tickets_by_tag.png", "TicketsByTag"),
        ("Tickets por Departamento", "charts_metrics/tickets_by_department.png", "TicketsByDepartment"),
        ("Tickets por Mês", "charts_metrics/tickets_by_month.png", "TicketsByMonth"),
    ]

    for titulo, path, metric_name in charts:

        story.append(Paragraph(f"<b>{titulo}:</b>", styles["Heading2"]))
        story.append(Spacer(1, 6))

        if metric_name != "TicketsByMonth":
            vals = extract_metric(data, metric_name)
            labels, values = prepare_chart_data(vals)
            if "Canal" in titulo or "Categoria" in titulo:
                plot_pie(labels, values, titulo, path)
            else:
                plot_bar(labels, values, titulo, path)
            for label, value in zip(labels, values):
                story.append(Paragraph(f"{label}: {value}", styles["Normal"]))

            story.append(Spacer(1, 10))
        story.append(Image(path, width=5*inch, height=3*inch))
        story.append(Spacer(1, 20))
    priority_data = qtd_tkt_priority["data"]

    resultados = plot_line_qtd_priority_month(priority_data)

    for item in resultados:
        story.append(Paragraph(f"<b>{item['titulo']}</b>", styles["Heading2"]))
        story.append(Spacer(1, 4))

        story.append(Paragraph(item["texto"], styles["Normal"]))
        story.append(Spacer(1, 12))

        story.append(Image(item["imagem"], width=5*inch, height=3*inch))
        story.append(Spacer(1, 25))

    status_data = qtd_tkt_status["data"]
    resultados2 = plot_line_qtd_status_month(status_data)

    for item in resultados2:
        story.append(Paragraph(f"<b>{item['titulo']}</b>", styles["Heading2"]))
        story.append(Spacer(1, 4))

        story.append(Paragraph(item["texto"], styles["Normal"]))
        story.append(Spacer(1, 12))

        story.append(Image(item["imagem"], width=5*inch, height=3*inch))
        story.append(Spacer(1, 25))


    doc.build(story)

    return FileResponse(
        pdf_filename,
        media_type="application/pdf",
        filename=pdf_filename,
    )
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
