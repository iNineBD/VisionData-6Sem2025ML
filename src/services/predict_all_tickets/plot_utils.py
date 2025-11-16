import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "models" / "all_tickets_kaggle"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def plot_predictions(train, test, pred, name):
    plt.figure(figsize=(15, 7))
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
