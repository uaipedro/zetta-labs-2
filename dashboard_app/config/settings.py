"""
Configurações centralizadas do dashboard
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = BASE_DIR / "reports"
PROCESSED_DIR = REPORTS_DIR / "processed"
MODELS_DIR = REPORTS_DIR / "models"
CHECKPOINTS_DIR = REPORTS_DIR / "checkpoints"
DATA_DIR = BASE_DIR / "data"

# Cache
CACHE_TTL = 1800  # 30 minutos

# Visualização
CHART_HEIGHT = 400
MAP_HEIGHT = 500
COLORSCALES = {
    'deforestation': 'Reds',
    'social': 'Greens',
    'economic': 'Blues',
    'enforcement': 'Oranges'
}

# Página
PAGE_CONFIG = {
    "page_title": "Dashboard Desmatamento Pará - Análise Avançada",
    "page_icon": "🌳",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# KPIs principais
KEY_METRICS = [
    "total_desmatamento",
    "taxa_crescimento",
    "municipios_criticos",
    "efetividade_fiscalizacao"
]

# Features importantes identificadas
IMPORTANT_FEATURES = [
    'desmatamento_lag1',
    'desmatamento_ma3', 
    'ibama_autos_flora',
    'Índice de Progresso Social',
    'PIB per capita 2021',
    'periodo_presidencial_encoded'
] 