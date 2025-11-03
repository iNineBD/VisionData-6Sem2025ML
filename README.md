# 6Sem2025ML

Repositório para o projeto de Machine Learning do 6º semestre de 2025.

## Funcionamento dos modelos

O ciclo de desenvolvimento dos modelos neste projeto segue as melhores práticas de MLOps e ciência de dados, com etapas bem definidas e rastreáveis:

### 1. Modelos para Diferentes Alvos

- **Todos os tickets:** Um modelo global prevê a quantidade total de tickets ao longo do tempo.
- **5 maiores companhias:** Para cada uma das 5 companhias com mais tickets registrados, são treinados modelos específicos.
- **5 maiores produtos:** Para cada um dos 5 produtos com mais tickets registrados, são treinados modelos dedicados.

### 2. Feature Store

- O projeto utiliza scripts de feature engineering para criar e armazenar variáveis derivadas relevantes para os modelos:

### Dicionário das Features de Engenharia de Dados

| Feature         | Descrição                                                                 |
|-----------------|--------------------------------------------------------------------------|
| `day`           | Dia do mês (1 a 31)                                                      |
| `month`         | Mês do ano (1 a 12)                                                      |
| `year`          | Ano da data                                                              |
| `weekday`       | Dia da semana (0=segunda, 6=domingo)                                     |
| `weekofyear`    | Número da semana no ano (1 a 52)                                         |
| `flag_1`        | Valor do ticket_count do dia anterior (lag de 1 dia)                     |
| `flag_2`        | Valor do ticket_count de 2 dias atrás                                    |
| `flag_3`        | Valor do ticket_count de 3 dias atrás                                    |
| `flag_7`        | Valor do ticket_count de 7 dias atrás (1 semana)                         |
| `flag_14`       | Valor do ticket_count de 14 dias atrás (2 semanas)                       |
| `flag_30`       | Valor do ticket_count de 30 dias atrás (1 mês)                           |
| `roll_mean_3`   | Média móvel dos últimos 3 dias (excluindo o dia atual)                   |
| `roll_std_3`    | Desvio padrão dos últimos 3 dias                                         |
| `roll_mean_7`   | Média móvel dos últimos 7 dias                                           |
| `roll_std_7`    | Desvio padrão dos últimos 7 dias                                         |
| `roll_mean_14`  | Média móvel dos últimos 14 dias                                          |
| `roll_std_14`   | Desvio padrão dos últimos 14 dias                                        |
| `roll_mean_30`  | Média móvel dos últimos 30 dias                                          |
| `roll_std_30`   | Desvio padrão dos últimos 30 dias                                        |
| `diff_1`        | Diferença do ticket_count em relação ao dia anterior                     |
| `pct_change_1`  | Variação percentual do ticket_count em relação ao dia anterior           |
| `is_month_start`| 1 se for o primeiro dia do mês, 0 caso contrário                         |
| `is_month_end`  | 1 se for o último dia do mês, 0 caso contrário                           |

- As features são salvas em arquivos processados para reuso e para servir a API.

### 3. Tratamento dos Dados

- Os dados brutos são carregados, limpos e pré-processados (remoção de nulos, padronização de datas, filtragem de outliers, etc).

#### Exemplos de uso das funções de tratamento de dados

```python
# remoção de nulos
df = df.dropna()
# conversão de datas
df['date'] = pd.to_datetime(df['date'])
# filtragem de outliers
df = df[(df['ticket_count'] >= lower_bound) & (df['ticket_count'] <= upper_bound)]
```

### 4. Variáveis de Treino e Teste

- Os dados são divididos em conjuntos de treino e teste (tipicamente 80% treino, 20% teste, ou conforme o script).

```python
# Dividir em treino e teste
split_idx = int(len(df_prophet) * (1 - 0.2))
train, test = df_prophet.iloc[:split_idx], df_prophet.iloc[split_idx:]
```

- Para séries temporais, a divisão respeita a ordem temporal para evitar vazamento de informação.

### 5. Padronização

As variáveis numéricas podem ser padronizadas/normalizadas conforme a necessidade do modelo.

#### Exemplo de padronização com StandardScaler

```python
from sklearn.preprocessing import StandardScaler

# dataframe com features numéricas
scaler = StandardScaler()
cols_to_scale = [
  'flag_1', 'flag_2', 'flag_3', 'flag_7', 'flag_14', 'flag_30',
  'roll_mean_3', 'roll_std_3', 'roll_mean_7', 'roll_std_7',
  'roll_mean_14', 'roll_std_14', 'roll_mean_30', 'roll_std_30',
  'diff_1', 'pct_change_1'
]
df_features[cols_to_scale] = scaler.fit_transform(df_features[cols_to_scale])
```

O pipeline de features garante que a transformação seja aplicada de forma consistente.

### 6. Treinamento com Diferentes Algoritmos

- Para o treinamento dos modelos, são utilizados diferentes algoritmos conforme o caso:

  - **LightGBM:** Um modelo de boosting eficiente para regressão.
  - **SARIMAX:** Modelo estatístico para séries temporais com componentes sazonais.
  - **Prophet:** Modelo de séries temporais desenvolvido pelo Facebook, adequado para dados com sazonalidade e feriados.

- Para cada alvo (todos os tickets, companhias, produtos), foram utilizados os modelos de acordo com a necessidade.
  - Todos os tickets: Prophet
  - Companhias: LightGBM e SARIMAX
  - Produtos: LightGBM e SARIMAX

### 7. Métricas de Desempenho

- Os modelos são avaliados com métricas apropriadas, como:
  - **MSE** (Erro Quadrático Médio)
  - **MAE** (Erro Absoluto Médio)
  - **RMSE** (Raiz do Erro Quadrático Médio)
  - **R²** (Coeficiente de Determinação)

- Métricas de erro são calculadas no conjunto de teste para avaliar a performance dos modelos, quanto menor o erro, melhor o modelo.

- Métrica R² indica a proporção da variância explicada pelo modelo, quanto mais próximo de 1, melhor o ajuste.

### 8. Registro no MLflow

Todos os experimentos, métricas e modelos treinados são registrados no MLflow.

Para cada run, são salvos:

- O modelo treinado (LightGBM, SARIMAX ou Prophet)
- As métricas de desempenho
- Os principais hiperparâmetros do modelo

Isso permite rastreabilidade, comparação e reprodutibilidade dos experimentos.

## Pre-commit hooks

Este repositório usa pre-commit para lint/format e validação de mensagens de commit.

Instalação local (recomendado executar no ambiente virtual):

```bash
pip install --user pre-commit
pre-commit install
pre-commit install --hook-type commit-msg   # instala hook de commit-msg se houver
pre-commit run --all-files                   # executa todos os hooks uma vez
```

## Estrutura de Diretórios para Projeto de Machine Learning

```sh
├── .github/                      # Configurações GitHub Actions (CI/CD)
│   └── workflows/
│       ├── cd.yml                # Pipeline de Continuous Deployment
│       └── ci.yml                # Pipeline de Continuous Integration (lint, testes, SonarQube)
├── data/                         # Diretório para datasets e arquivos de dados
│   └── .gitkeep
├── models/                       # Diretório para modelos ML treinados (.pkl, .h5, etc.)
│   └── .gitkeep
├── src/                          # Código-fonte principal da aplicação
│   ├── services/                 # Módulos de serviços e lógica de negócio
│   │   └── serviceHello.py       # Serviço de exemplo
│   └── main.py                   # Ponto de entrada da aplicação
├── tests/                        # Testes unitários e de integração
│   └── test_example.py           # Testes do serviceHello
├── .gitignore                    # Arquivos/diretórios ignorados pelo Git
├── README.md                     # Documentação do projeto
├── requirements.txt              # Dependências Python do projeto
├── setup.py                      # Configuração de empacotamento do projeto
└── sonar-project.properties      # Configuração SonarQube para análise de código
```
