FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema necessárias
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# Copiar arquivos
COPY . .

# Instalar dependências Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Comando de inicialização: treina e depois inicia o servidor
CMD ["bash", "-c", "python src/services/predict_all_ticketsv2/2_train_all_tickets.py && python controller/tickets_controller.py"]
