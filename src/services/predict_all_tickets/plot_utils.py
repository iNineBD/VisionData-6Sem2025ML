import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "models" / "all_tickets_kaggle"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def plot_predictions(train, test, pred, name):
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label="Treino", color="blue", linestyle="-")
    plt.plot(test.index, test, label="Teste", color="green", marker=".", linestyle="-")
    plt.plot(
        test.index, pred, label="Previsto", color="red", marker=".", linestyle="--"
    )
    plt.title(f"Histórico, Teste e Previsão - {name}", fontsize=16)
    plt.xlabel("Data", fontsize=12)
    plt.ylabel("Contagem de Tickets", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt_path = MODELS_DIR / f"{name}_forecast.png"
    plt.savefig(plt_path)
    plt.close()
    return plt_path

def plot_total_forecast_image(historical_df, predictions_df, title="Total - Previsão Geral"):
    plt.figure(figsize=(12, 6))

    plt.plot(
        historical_df["date"],
        historical_df["ticket_count"],
        label="Histórico",
        color="#3a0ca3",
    )

    plt.plot(
        predictions_df["date"],
        predictions_df["ticket_count"],
        label="Previsão",
        color= "#ff5ac8",
        linestyle="--"
    )

    plt.title(title)
    plt.xlabel("Data")
    plt.ylabel("Tickets")
    plt.grid(True)
    plt.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)

    return buffer.getvalue()
