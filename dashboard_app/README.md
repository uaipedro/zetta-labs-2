# Dashboard de Análise do Desmatamento no Pará v2.0

## 🌳 Visão Geral

Dashboard interativo e componentizado para análise avançada do desmatamento no estado do Pará, com insights acionáveis e previsões baseadas em Machine Learning.

## 🚀 Características Principais

### 📊 Resumo Executivo

- **KPIs principais** com visualização em cards
- **Insights automáticos** baseados nos dados
- **Top municípios críticos** com visualizações
- **Recomendações estratégicas** personalizadas

### 🗺️ Análise Espacial

- **Mapas coropléticos** interativos
- **Comparação visual** entre desmatamento e indicadores sociais
- **Análise de correlação espacial**
- **Modos de visualização**: absoluto e percentual

### 📈 Análise Temporal

- **Tendências históricas** com anotações
- **Análise por período presidencial**
- **Simulador de cenários** interativo
- **Previsões para 2024-2025**

### 🤖 Análise de Modelos

- **Comparação de performance** (Random Forest vs XGBoost)
- **Feature importance** detalhada
- **Análise de resíduos** e diagnósticos
- **Recomendações de melhorias**

## 📁 Estrutura do Projeto

```
dashboard_app/
├── app.py                 # Aplicação principal
├── config/
│   ├── __init__.py
│   └── settings.py        # Configurações centralizadas
├── data/
│   ├── __init__.py
│   └── loader.py          # Carregamento de dados com cache
├── utils/
│   ├── __init__.py
│   ├── metrics.py         # Cálculo de métricas e KPIs
│   ├── visualizations.py  # Visualizações reutilizáveis
│   └── predictions.py     # Motor de previsões
└── components/
    ├── __init__.py
    ├── executive_summary.py  # Componente de resumo executivo
    ├── spatial_analysis.py   # Componente de análise espacial
    ├── temporal_analysis.py  # Componente de análise temporal
    └── model_analysis.py     # Componente de análise de modelos
```

## 🛠️ Como Executar

### Opção 1: Script de Execução

```bash
python run_dashboard_v2.py
```

### Opção 2: Streamlit Direto

```bash
streamlit run dashboard_app/app.py
```

## 📊 Dados Necessários

O dashboard espera os seguintes arquivos:

- `reports/processed/analytical_base_table.parquet`
- `reports/processed/model_comparison_data.parquet` (opcional)
- `reports/checkpoints/all_results_*.pkl` (modelos)
- `data/PA_Municipios_2024/PA_Municipios_2024.shp` (shapefile)

## 🎯 Insights de Valor

### Para Gestores Públicos

- Identificação de **hotspots** de desmatamento
- **ROI de políticas** de fiscalização
- **Cenários futuros** baseados em diferentes estratégias

### Para Pesquisadores

- **Análise de drivers** do desmatamento
- **Correlações** entre variáveis socioeconômicas
- **Performance de modelos** preditivos

### Para ONGs e Sociedade Civil

- **Transparência** nos dados ambientais
- **Monitoramento** de tendências
- **Base para advocacy** com dados sólidos

## 🔧 Personalização

### Adicionar Novos Indicadores

1. Atualizar `get_available_variables()` em `spatial_analysis.py`
2. Adicionar lógica de cálculo em `metrics.py`
3. Criar visualização em `visualizations.py`

### Modificar Cenários

1. Editar scenarios dict em `temporal_analysis.py`
2. Ajustar pesos em `predictions.py`

## 📈 Melhorias Futuras

- [ ] Integração com dados em tempo real
- [ ] Exportação de relatórios PDF
- [ ] API REST para integração
- [ ] Dashboard mobile-friendly
- [ ] Análise de sentimento de notícias

## 👥 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o repositório
2. Crie sua feature branch
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob licença MIT. Veja o arquivo LICENSE para detalhes.

---

Desenvolvido com ❤️ por Zetta Labs
