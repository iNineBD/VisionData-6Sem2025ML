import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from datetime import datetime, timedelta

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
