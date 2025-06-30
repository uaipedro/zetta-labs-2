"""
Componente de Análise de Modelos
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Optional

# Configurar path para imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from utils.metrics import MetricsCalculator
from utils.visualizations import Visualizer
from utils.predictions import PredictionEngine


def render_model_analysis(model_results: Optional[pd.DataFrame], 
                          feature_importance: Optional[pd.DataFrame],
                          best_result: Optional[pd.Series]) -> None:
    """Renderiza a análise dos modelos de Machine Learning"""
    
    st.markdown("### 🤖 Análise de Modelos")

    if model_results is None or model_results.empty:
        st.warning("Não foi possível carregar os resultados dos modelos. Análise indisponível.")
        return

    # Layout principal
    with st.container():
        # Visão geral da performance
        st.markdown("### 📊 Performance dos Modelos")
        
        if model_results is not None and not model_results.empty:
            col1, col2 = st.columns(2)
            
            
            with col1:
                st.markdown("##### 📈 Evolução da Performance")
                
                # Gráfico de evolução do R²
                fig_r2 = Visualizer.create_trend_chart(
                    df=model_results,
                    x_col='test_year',
                    y_col='r2',
                    color_col='model_type',
                    title="Evolução do R² (R-squared) ao Longo do Tempo",
                    y_axis_title="R² (R-squared)",
                    hover_template='%{y:.3f}'
                )
                st.plotly_chart(fig_r2, use_container_width=True, key="r2_evolution_chart")
            
            with col2:
                st.markdown("##### 🔍 Distribuição do Erro")
                
                # Boxplot do RMSE
                fig_rmse = Visualizer.create_error_distribution_chart(
                    data=model_results,
                    x='model_type',
                    y='rmse',
                    title="Distribuição do RMSE por Modelo",
                    color='model_type'
                )
                st.plotly_chart(fig_rmse, use_container_width=True, key="rmse_dist_chart")
            
            if feature_importance is not None:
                st.markdown("##### 🎯 Importância das Features")
                
                # Calcular importância média
                avg_importance = feature_importance.groupby('feature')['importance'].mean()
                avg_importance = avg_importance.sort_values(ascending=False).head(10)
                
                # Gráfico de importância
                fig_feat = Visualizer.create_feature_importance_chart(
                    feature_importance=avg_importance,
                    title="Top 10 Features (Média Geral)"
                )
                st.plotly_chart(fig_feat, use_container_width=True, key="feat_imp_chart")
            
            # Análise detalhada
            with st.expander("📖 Análise Detalhada dos Modelos"):
                # Métricas por modelo
                metrics_by_model = model_results.groupby('model_type').agg({
                    'r2': ['mean', 'std'],
                    'rmse': ['mean', 'std'],
                    'mae': ['mean', 'std']
                }).round(3)
                
                metrics_by_model.columns = [
                    'R² Médio', 'R² Desvio',
                    'RMSE Médio', 'RMSE Desvio',
                    'MAE Médio', 'MAE Desvio'
                ]
                
                st.dataframe(metrics_by_model, use_container_width=True)
                
                # Análise temporal
                st.markdown("##### 📈 Evolução Temporal das Métricas")
                
                # Gráficos separados para R² e Erros para melhor visualização
                col1, col2 = st.columns(2)

                with col1:
                    fig_r2 = Visualizer.create_trend_chart(
                        df=model_results,
                        x_col='test_year',
                        y_col='r2',
                        color_col='model_type',
                        title="Evolução do R² (R-squared)",
                        y_axis_title="R²"
                    )
                    st.plotly_chart(fig_r2, use_container_width=True, key="r2_perf_chart")
                
                with col2:
                    fig_errors = Visualizer.create_metrics_evolution_chart(
                        data=model_results,
                        metrics=['rmse', 'mae'],
                        x='test_year',
                        title="Evolução das Métricas de Erro"
                    )
                    st.plotly_chart(fig_errors, use_container_width=True, key="errors_perf_chart")

                # Análise de erros
                st.markdown("##### 🔍 Análise de Erros")
                
                # Distribuição dos erros
                fig_errors = Visualizer.create_error_distribution_chart(
                    data=model_results,
                    x='model_type',
                    y='rmse',
                    title="Distribuição do RMSE por Modelo",
                    color='model_type'
                )
                st.plotly_chart(fig_errors, use_container_width=True)

                st.markdown("##### 🔍 Distribuição do Erro (MAE)")
                
                # Boxplot do MAE
                fig_mae = Visualizer.create_error_distribution_chart(
                    data=model_results,
                    x='model_type',
                    y='mae',
                    title="Distribuição do MAE por Modelo",
                    color='model_type'
                )
                st.plotly_chart(fig_mae, use_container_width=True, key="mae_dist_chart")

                # Análise do melhor modelo
                if best_result is not None:
                    st.markdown("---")
                    st.markdown("### 🏆 Análise do Melhor Modelo")

                    best_model_name = "XGBoost" if best_result['model_type'] == 'xgboost' else "Random Forest"
                    st.success(f"O melhor desempenho geral foi do **{best_model_name}** no ano de teste de **{int(best_result['test_year'])}**, com um **R² de {best_result['r2']:.3f}**.")
        else:
            st.warning("Dados de performance dos modelos não disponíveis")
        
        