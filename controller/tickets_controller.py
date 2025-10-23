import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import uvicorn
import os
from datetime import datetime, timedelta
from src.services.predict_company.app import run_pipeline
from src.config import config
app = FastAPI(title="Tickets Prediction API", version="1.0.0")

# model all tickets
all_tickets_model_name = "Auto-ARIMA"
all_tickets_model_path = f"./models/{all_tickets_model_name}_model.pkl"

try:
    with open(all_tickets_model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Modelo {all_tickets_model_name} carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    model = None


class PredictionResponse(BaseModel):
    model_name: str
    days: int
    predictions: dict


@app.get("/predict", response_model=PredictionResponse)
def predict_default():
    """Retorna previsão para os próximos 30 dias"""

    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado")

    try:
        # Fazer previsão
        predictions = model.forecast(steps=30)

        # Gerar datas futuras
        start_date = datetime.now().date()
        dates = [start_date + timedelta(days=i) for i in range(1, 30 + 1)]

        # Converter para dicionário: {"2025-10-15": 100, "2025-10-16": 105, ...}
        predictions_dict = {
            date.strftime("%Y-%m-%d"): int(round(pred))
            for date, pred in zip(dates, predictions)
        }

        return {
            "model_name": all_tickets_model_name,
            "days": 30,
            "predictions": predictions_dict,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na previsão: {str(e)}")
    
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
