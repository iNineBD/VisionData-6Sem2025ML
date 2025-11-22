FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app

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
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]
