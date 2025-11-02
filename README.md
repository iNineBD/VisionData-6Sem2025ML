# 6Sem2025ML

Repositório para o projeto de Machine Learning do 6º semestre de 2025.

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
