# 6Sem2025ML

## Pre-commit hooks

Este repositório usa pre-commit para lint/format e validação de mensagens de commit.

Instalação local (recomendado executar no ambiente virtual):

```bash
pip install --user pre-commit
pre-commit install
pre-commit install --hook-type commit-msg   # instala hook de commit-msg se houver
pre-commit run --all-files                   # executa todos os hooks uma vez
```

CI:

- Execute `pre-commit run --all-files --show-diff-on-failure` antes dos testes para garantir consistência.

## Estrutura de Diretórios para Projeto de Machine Learning

.
├── data/                # Dados brutos e processados usados nos experimentos (não versionar arquivos grandes)
├── dist/                # Artefatos de distribuição gerados (wheel / sdist) após build
├── models/              # Modelos treinados e checkpoints serializados (.pkl, .pt, .joblib, etc.)
├── README.md            # Este arquivo — documentação do projeto e estrutura
├── requirements.txt     # Lista de dependências (pip install -r requirements.txt)
├── setup.py             # Configuração de empacotamento do projeto (pip install -e . / build)
├── sonar-project.properties
|     # Configuração do SonarQube/SonarCloud (paths, cobertura, chave do projeto)
├── src/                 # Código-fonte principal do projeto (package root)
│   ├── __init__.py      # Marca `src` como pacote Python (ponto de entrada do package)
│   ├── main.py          # Script simples / entrypoint — funções de exemplo e CLI mínima
│   └── services/        # Módulos de lógica aplicada (serviços, funções reutilizáveis)
│       └── serviceHello.py
│           # Funções utilitárias de exemplo (ex.: hello(), add()) usadas nos testes
└── tests/               # Testes automatizados (pytest)
    └── test_hello.py    # Testes simples para validar serviceHello / main
