# %%

# Análise de Chamados por Empresa
# Objetivo
# Analisar a quantidade de chamados abertos por diferentes empresas ao longo do tempo, identificando tendências e padrões.
# %%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# %%
# Carregar os dados

df = pd.read_csv(REPO_ROOT / "data" / "rows.csv")
df

# %%
# companys distincts
print(f"Total de empresas distintas: {df['Company'].nunique()}")

# %%
# quantida de tickets por company
tickets_per_company = df["Company"].value_counts()
print(f"Quantidade de tickets por empresa:\n{tickets_per_company.head(5)}")
# %%
# grafico de barras das 5 maiores empresas com mais tickets
top_5_companies = tickets_per_company.head(5)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_5_companies.index, y=top_5_companies.values, palette="viridis")
plt.title("Top 5 Empresas com Mais Tickets")
plt.xlabel("Empresa")
plt.ylabel("Quantidade de Tickets")
plt.show()

# %%

# coluna Date received para datetime e setar como index
df["Date received"] = pd.to_datetime(df["Date received"])
df.set_index("Date received", inplace=True)

# %%

# select somente os tickets com as top 5 companies
top_5_company_names = top_5_companies.index.tolist()
df_top_5 = df[df["Company"].isin(top_5_company_names)]

# %%
# agrupar por month e company e contar a quantidade de tickets
tickets_monthly = (
    df_top_5.groupby([pd.Grouper(freq="M"), "Company"])
    .size()
    .reset_index(name="Ticket Count")
)
tickets_monthly
# %%
# plotar grafico de linhas da quantidade de tickets por month para as top 5 companies
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=tickets_monthly, x="Date received", y="Ticket Count", hue="Company", marker="o"
)
plt.title("Quantidade de Tickets Mensais por Empresa (Top 5)")
plt.xlabel("Mês")
plt.ylabel("Quantidade de Tickets")
plt.legend(title="Empresa")
plt.show()
# %%
# Análise de tendências
# Calcular a média móvel de 3 meses para suavizar as tendências
tickets_monthly["3-Month MA"] = tickets_monthly.groupby("Company")[
    "Ticket Count"
].transform(lambda x: x.rolling(window=3).mean())
# Plotar gráfico com média móvel
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=tickets_monthly, x="Date received", y="3-Month MA", hue="Company", marker="o"
)
plt.title("Média Móvel de 3 Meses da Quantidade de Tickets por Empresa (Top 5)")
plt.xlabel("Mês")
plt.ylabel("Média Móvel de Tickets")
plt.legend(title="Empresa")
plt.show()
# %%
