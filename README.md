# 6Sem2025ML

Repositório para o projeto de Machine Learning do 6º semestre de 2025.

## Funcionamento dos modelos

O ciclo de desenvolvimento dos modelos neste projeto segue as melhores práticas de MLOps e ciência de dados, com etapas bem definidas e rastreáveis:

### 1. Modelos para Diferentes Alvos

- **Todos os tickets:** Um modelo global prevê a quantidade total de tickets ao longo do tempo.
- **5 maiores companhias:** Para cada uma das 5 companhias com mais registros, são treinados modelos específicos.
- **5 maiores produtos:** Para cada um dos 5 produtos mais frequentes, são treinados modelos dedicados.

### 2. Feature Store

- O projeto utiliza scripts de feature engineering para criar e armazenar variáveis derivadas relevantes para os modelos (lags, médias móveis, variáveis sazonais, etc).
- As features são salvas em arquivos processados para reuso e para servir a API.

### 3. Tratamento dos Dados

- Os dados brutos são carregados, limpos e pré-processados (remoção de nulos, padronização de datas, filtragem de outliers, etc).
- O pipeline de preparação garante consistência entre treino, teste e produção.

### 4. Variáveis de Treino e Teste

- Os dados são divididos em conjuntos de treino e teste (tipicamente 70% treino, 30% teste, ou conforme o script).
- Para séries temporais, a divisão respeita a ordem temporal para evitar vazamento de informação.

### 5. Padronização

- As variáveis numéricas podem ser padronizadas/normalizadas conforme a necessidade do modelo.
- O pipeline de features garante que a transformação seja aplicada de forma consistente.

### 6. Treinamento com Diferentes Algoritmos

- Para cada alvo (todos os tickets, companhias, produtos), são treinados dois algoritmos principais:
  - **LightGBM:** Modelo de boosting para previsão tabular.
  - **SARIMAX:** Modelo estatístico para séries temporais.
- Para todos os tickets, é utilizado também o **Prophet** para previsão temporal agregada.

### 7. Métricas de Desempenho

- Os modelos são avaliados com métricas apropriadas, como:
  - **MSE** (Erro Quadrático Médio)
  - **MAE** (Erro Absoluto Médio)
  - **RMSE** (Raiz do Erro Quadrático Médio)
  - **R²** (Coeficiente de Determinação)

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
