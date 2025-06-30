# Desafio Zetta Labs 2025: Análise Preditiva de Desmatamento

Este projeto foi desenvolvido por Pedro Mambelli Fernandes como parte do desafio Zetta Labs 2025. O objetivo é realizar uma análise detalhada sobre o desmatamento, utilizando dados públicos para construir modelos preditivos capazes de identificar os municípios com maior risco de aumento nas taxas de desmatamento.

## Estrutura de Pastas

O projeto está organizado da seguinte forma para garantir clareza e reprodutibilidade:

```
zetta-labs-2/
├── dashboard_app/      # Módulos da aplicação do dashboard
│   ├── components/     # Componentes visuais do dashboard
│   ├── config/         # Configurações da aplicação
│   ├── data/           # Funções para carregamento de dados no dashboard
│   └── utils/          # Funções utilitárias para o dashboard
├── data/               # Dados brutos e processados
│   ├── ibama/          # Dados de infrações do IBAMA
│   └── PA_Municipios_2024/ # Shapefiles dos municípios do Pará
├── notebooks/          # Notebooks para análises e exploração
├── reports/            # Arquivos gerados pelas análises (relatórios, figuras, modelos)
│   ├── figures/        # Gráficos e visualizações
│   ├── models/         # Modelos de machine learning treinados
│   ├── processed/      # Dados intermediários e tabelas analíticas
│   └── results/        # Resultados dos experimentos e métricas
├── scripts/            # Scripts Python para o pipeline de dados e modelagem
│   ├── models/         # Definições das classes dos modelos (XGBoost, RandomForest)
│   ├── assess.py       # Avaliação e análise de performance dos modelos
│   ├── explore.py      # Análise exploratória dos dados
│   ├── modify.py       # Pré-processamento e engenharia de features
│   ├── model.py        # Orquestração do treinamento e predição
│   └── sampling.py     # Lógica para amostragem de dados
├── .gitignore
├── dashboard.py        # Scripts para execução do dashboard
├── main.py             # Script principal para executar o pipeline completo
├── pyproject.toml      # Metadados e dependências do projeto
└── requirements.txt    # Dependências do projeto para pip
```

## Como Executar o Projeto

Para reproduzir as análises e visualizar os resultados, siga os passos abaixo.

### Pré-requisitos

- Python 3.9+
- `pip` para gerenciamento de pacotes

### Instalação

1. Clone o repositório:

   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd zetta-labs-2
   ```

2. Crie um ambiente virtual (recomendado):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # No Windows: .venv\Scripts\activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

### Executando as Análises

## Análise Detalhada

Para uma compreensão passo a passo de toda a metodologia, desde a coleta e tratamento dos dados até a interpretação dos modelos, consulte o notebook explicativo:

- [`notebooks/analise_passo_a_passo.ipynb`](./notebooks/analise_passo_a_passo.ipynb)

Este notebook utiliza as funções e classes desenvolvidas nos scripts para demonstrar o processo.

Os resultados, incluindo modelos treinados, tabelas analíticas e figuras, serão salvos no diretório `/reports`.

### Executando o Dashboard

O projeto inclui um dashboard interativo para explorar os resultados das predições. Para iniciá-lo, execute o seguinte comando:

```bash
python run_dashboard.py
```

O dashboard estará disponível em seu navegador no endereço `http://localhost:8501`.
