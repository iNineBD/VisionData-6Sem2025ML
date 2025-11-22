import requests
import os
import matplotlib.pyplot as plt
PALETTE = [
    "#ff9ce6",  
    "#ff5ac8",  
    "#c77dff",  
    "#9d4edd",  
    "#7b2cbf",  
    "#5a4fcf",  
    "#4d6aff",  
    "#3a0ca3",      
    "#4361ee",  
]
BASE_URL = os.getenv("TARGET_API_URL")
TARGET_USER_EMAIL = os.getenv("TARGET_USER_EMAIL")
TARGET_USER_PASSWORD = os.getenv("TARGET_USER_PASSWORD")
def login(base_url: str, email: str, password: str) -> str:
    url = f"{base_url}/auth/login"

    payload = {
        "email": email,
        "login_type": "password",
        "microsoft_id_token": "",
        "password": password
    }

    response = requests.post(url, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"Erro no login: {response.status_code} {response.text}")

    data = response.json()
    
    if not data.get("success"):
        raise Exception("Login falhou")

    return data["data"]["token"]

def get_with_token(base_url: str, endpoint: str, token: str):
    url = f"{base_url}{endpoint}"

    headers = {
        "Authorization": f"Bearer {token}"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 401:
        raise Exception("Token expirado ou inválido")

    if response.status_code >= 400:
        raise Exception(f"Erro ao acessar {endpoint}: {response.status_code} {response.text}")

    return response.json()

def get_tickets(base_url, token):
    return get_with_token(base_url, "/metrics/tickets", token)

def get_mean_resolution(base_url, token):
    return get_with_token(base_url, "/metrics/tickets/mean-time-resolution-by-priority", token)

def get_qtd_by_month(base_url, token):
    return get_with_token(base_url, "/metrics/tickets/qtd-tickets-by-month", token)

def get_qtd_tickets_by_priority_year_month(base_url, token):
    return get_with_token(base_url, "/metrics/tickets/qtd-tickets-by-priority-year-month", token)

def get_qtd_tickets_by_status_year_month(base_url, token):
    return get_with_token(base_url, "/metrics/tickets/qtd-tickets-by-status-year-month", token)

token = login(BASE_URL, TARGET_USER_EMAIL, TARGET_USER_PASSWORD)

# print("TOKEN:", token)

tickets = get_tickets(BASE_URL, token)
data = tickets["data"]
total_tickets = data["totalTickets"]
# print(f"Tickets:", total_tickets)
# print("----------------------")
mean_res = get_mean_resolution(BASE_URL, token)
# print(f"Mean Resolution:", mean_res)
# print("----------------------")
qtd_month = get_qtd_by_month(BASE_URL, token)
# print("Qtd por mês:", qtd_month)
# print("----------------------")
qtd_tkt_priority = get_qtd_tickets_by_priority_year_month(BASE_URL, token)
# print("Qtd por mês:", qtd_tkt_priority)
# print("----------------------")
qtd_tkt_status = get_qtd_tickets_by_status_year_month(BASE_URL, token)
# print("Qtd por mês:", qtd_tkt_status)
# print("----------------------")
tickets

def extract_metric(data, metric_name):
    """
    Procura no array data["metrics"] o item com name == metric_name.
    Retorna a lista de valores, ex: [{ "name": "...", "value": ... }]
    """
    if "metrics" not in data:
        raise Exception("Campo 'metrics' não encontrado no data")

    for metric in data["metrics"]:
        if metric.get("name") == metric_name:
            return metric.get("values", [])

    raise Exception(f"Métrica '{metric_name}' não encontrada!")

def prepare_chart_data(values):
    labels = []
    numbers = []

    for item in values:
        label = item.get("name")
        number = item.get("value")

        if label == "N/A":
            continue  # ignora

        labels.append(label)
        numbers.append(number)

    return labels, numbers

def plot_pie(labels, values, title, output_path):
    plt.figure(figsize=(6, 4))
    colors = PALETTE[:len(values)] 
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=colors)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_bar(labels, values, title, output_path):
    plt.figure(figsize=(8, 5))
    colors = PALETTE[:len(labels)]
    plt.bar(labels, values, color=colors)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Quantidade")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_all_charts(tickets_data, qtd_month_data):
    data = tickets_data["data"]

    # ---- TicketsByChannel (pizza)
    channel_values = extract_metric(data, "TicketsByChannel")
    labels, values = prepare_chart_data(channel_values)
    plot_pie(labels, values, "Tickets por Canal", "charts/tickets_by_channel.png")

    # ---- TicketsByCategory (pizza)
    cat_values = extract_metric(data, "TicketsByCategory")
    labels, values = prepare_chart_data(cat_values)
    plot_pie(labels, values, "Tickets por Categoria", "charts/tickets_by_category.png")

    # ---- TicketsByTag (barra)
    tag_values = extract_metric(data, "TicketsByTag")
    tag_values = {k: v for k, v in tag_values.items() if k != "N/A"}
    labels, values = prepare_chart_data(tag_values)
    plot_bar(labels, values, "Tickets por Tag", "charts/tickets_by_tag.png")

    # ---- TicketsByDepartment (barra)
    dep_values = extract_metric(data, "TicketsByDepartment")
    labels, values = prepare_chart_data(dep_values)
    plot_bar(labels, values, "Tickets por Departamento", "charts/tickets_by_department.png")

    plot_line_qtd_month(qtd_month_data, "charts/tickets_by_month.png")

ORDERED_MONTHS = [
    "janeiro", "fevereiro", "marco", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"
]

def plot_line_qtd_month(qtd_month_data, output_path="charts/qtd_by_month.png"):
    data = qtd_month_data["data"]

    plt.figure(figsize=(10, 6))

    for idx, (year, values_list) in enumerate(data.items()):
        values_dict = values_list[0]

        months = ORDERED_MONTHS
        values = [values_dict.get(month, 0) for month in months]
        color = PALETTE[idx % len(PALETTE)]
        plt.plot(months, values, marker="o", label=year, linewidth=2.3, color=color)

    plt.title("Quantidade de Tickets por Mês")
    plt.xlabel("Mês")
    plt.ylabel("Quantidade")
    plt.xticks(rotation=45)
    plt.legend(title="Ano")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_line_qtd_priority_month(data_priority, output_dir="charts_priority"):
    """
    Gera um gráfico por ANO contendo todas as prioridades juntas.
    Cada prioridade aparece como uma linha no gráfico usando as cores da PALETTE.
    """
    os.makedirs(output_dir, exist_ok=True)

    resultados_pdf = []

    anos_disponiveis = set()
    for prioridade, anos in data_priority.items():
        anos_disponiveis.update(anos.keys())

    anos_disponiveis = sorted(list(anos_disponiveis))

    for ano in anos_disponiveis:
        plt.figure(figsize=(12, 5))

        total_ano_global = 0

        prioridades_ordenadas = list(data_priority.keys())

        for idx, prioridade in enumerate(prioridades_ordenadas):

            anos = data_priority[prioridade]
            if ano not in anos:
                continue

            valores_dict = anos[ano][0]

            meses = ORDERED_MONTHS
            valores = [valores_dict.get(m, 0) for m in meses]

            total_ano_global += sum(valores)

            cor = PALETTE[idx % len(PALETTE)]

            plt.plot(
                meses,
                valores,
                marker="o",
                linewidth=2,
                label=prioridade,
                color=cor
            )

        plt.title(f"Tickets por Prioridade • {ano}")
        plt.xlabel("Mês")
        plt.ylabel("Quantidade")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Prioridade")

        filename = f"{output_dir}/prioridades_{ano}.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        texto = (
            f"Análise consolidada de <b>todas as prioridades</b> no ano de <b>{ano}</b>:<br/>"
            f"Foram registrados <b>{total_ano_global}</b> tickets no total."
        )

        resultados_pdf.append({
            "ano": ano,
            "titulo": f"Prioridades - {ano}",
            "texto": texto,
            "imagem": filename
        })

    return resultados_pdf

def plot_line_qtd_status_month(data_status, output_dir="charts_status"):
    """
    Gera um gráfico por ANO contendo todas as linhas de status.
    Cada status usa uma cor da PALETTE.
    """

    os.makedirs(output_dir, exist_ok=True)

    resultados_pdf = []


    anos_disponiveis = set()
    for status, anos in data_status.items():
        anos_disponiveis.update(anos.keys())

    anos_disponiveis = sorted(list(anos_disponiveis))

    for ano in anos_disponiveis:
        plt.figure(figsize=(12, 5))

        total_ano_global = 0

        status_ordenados = list(data_status.keys())

        for idx, status in enumerate(status_ordenados):

            anos = data_status[status]
            if ano not in anos:
                continue

            valores_dict = anos[ano][0] 

            meses = ORDERED_MONTHS
            valores = [valores_dict.get(m, 0) for m in meses]

            total_ano_global += sum(valores)

            cor = PALETTE[idx % len(PALETTE)]

            plt.plot(
                meses,
                valores,
                marker="o",
                linewidth=2,
                label=status,
                color=cor
            )

        plt.title(f"Tickets por Status • {ano}")
        plt.xlabel("Mês")
        plt.ylabel("Quantidade")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Status")

        filename = f"{output_dir}/status_{ano}.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        texto = (
            f"Análise consolidada de <b>todos os status</b> no ano de <b>{ano}</b>:<br/>"
            f"Foram registrados <b>{total_ano_global}</b> tickets no total."
        )

        resultados_pdf.append({
            "ano": ano,
            "titulo": f"Status - {ano}",
            "texto": texto,
            "imagem": filename
        })

    return resultados_pdf
