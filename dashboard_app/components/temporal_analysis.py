"""
Componente de Análise Temporal
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Optional, List, Any

# Configurar path para imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from utils.metrics import MetricsCalculator
from utils.visualizations import Visualizer
from utils.predictions import PredictionEngine


def render_temporal_analysis(
    df: pd.DataFrame, 
    model: Optional[Any] = None, 
    metadata: Optional[Dict] = None,
    predictions_hist_df: Optional[pd.DataFrame] = None
) -> None:
    """Renderiza a análise temporal do dashboard"""
    
    st.markdown("### 📈 Evolução Temporal")
    
    # Gerar previsões futuras se modelo disponível
    future_predictions_df = None
    if model is not None and 'ano' in df.columns:
        latest_data = df[df['ano'] == df['ano'].max()].copy()
        pred_2024 = PredictionEngine.make_predictions(model, latest_data, {"base": {}})
        
        if pred_2024 and "base" in pred_2024:
            pred_2024_val = pred_2024["base"]
            pred_2025_val = pred_2024_val * 0.98  # Ajuste conservador
            
            future_predictions_df = pd.DataFrame({
                'ano': [2024, 2025],
                'desmatamento_km2': [pred_2024_val, pred_2025_val]
            })
    
    # Layout compacto
    col_chart, col_stats = st.columns([2, 1])
    
    with col_chart:
        # Mostrar previsões futuras
        if future_predictions_df is not None and not future_predictions_df.empty:
            pred_2024_str = f"{future_predictions_df.iloc[0]['desmatamento_km2']:.0f} km²"
            pred_2025_str = f"{future_predictions_df.iloc[1]['desmatamento_km2']:.0f} km²"
            st.info(f"🔮 **Previsões Futuras:** 2024 = {pred_2024_str} | 2025 = {pred_2025_str}")
        
        # Preparar dados para o gráfico
        yearly_totals = df.groupby('ano')['desmatamento_km2'].sum().reset_index()

        # Preparar predições históricas anuais
        historical_preds_yearly = None
        if predictions_hist_df is not None and 'ano' in predictions_hist_df.columns and 'prediction' in predictions_hist_df.columns:
            historical_preds_yearly = predictions_hist_df.groupby('ano')['prediction'].sum().reset_index()
            historical_preds_yearly = historical_preds_yearly.rename(columns={'prediction': 'desmatamento_km2'})

        # Gráfico temporal
        timeline_fig = Visualizer.create_trend_chart(
            df=yearly_totals, 
            x_col='ano', 
            y_col='desmatamento_km2',
            title="Evolução Anual do Desmatamento (Real vs. Previsto)",
            predictions_df=future_predictions_df,
            historical_predictions_df=historical_preds_yearly,
            y_axis_title="Desmatamento (km²)",
            hover_template='%{y:,.0f} km²'
        )
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
    
    with col_stats:
        st.markdown("#### 📊 Estatísticas")
        
        # Calcular estatísticas da tendência
        yearly_totals = df.groupby('ano')['desmatamento_km2'].sum()
        
        # Tendência recente
        recent_years_df = yearly_totals.tail(5).reset_index()
        if len(recent_years_df) >= 2:
            start_year, end_year = int(recent_years_df['ano'].min()), int(recent_years_df['ano'].max())
            start_val = recent_years_df.iloc[0, 1]
            end_val = recent_years_df.iloc[-1, 1]
            
            recent_trend = (end_val - start_val) / start_val * 100 if start_val != 0 else 0
            trend_direction = "📈 Crescente" if recent_trend > 0 else "📉 Decrescente"
            
            st.metric(f"Tendência ({start_year}–{end_year})", trend_direction, f"{recent_trend:.1f}%")
        
        # Estatísticas essenciais
        st.metric("Máximo Anual", f"{yearly_totals.max():.0f} km²")
        st.metric("Mínimo", f"{yearly_totals.min():.0f} km²")
        st.metric("Média", f"{yearly_totals.mean():.0f} km²")
        
        # Anos extremos
        worst_year = yearly_totals.idxmax()
        best_year = yearly_totals.idxmin()
        
        st.markdown(f"**🔝 Pior:** {worst_year}")
        st.markdown(f"**✅ Melhor:** {best_year}")
    
    # Análise por períodos
    if 'periodo_presidencial' in df.columns:
        with st.expander("🗓️ Análise por Períodos Presidenciais", expanded=True):
            col_graf, col_tab = st.columns(2)
            
            with col_graf:
                periodo_stats = df.groupby('periodo_presidencial').agg(
                    total_desmatado=('desmatamento_km2', 'sum'),
                    media_anual=('desmatamento_km2', 'mean'),
                    anos_no_periodo=('ano', 'nunique')
                ).reset_index()
                
                # Gráfico compacto
                periodo_dict = periodo_stats.set_index('periodo_presidencial')['media_anual'].to_dict()
                fig_periodo = Visualizer.create_scenario_comparison(
                    scenarios=periodo_dict,
                    title="Média de Desmatamento por Período"
                )
                st.plotly_chart(fig_periodo, use_container_width=True)
            
            with col_tab:
                # Tabela compacta
                periodo_summary = periodo_stats.round(0)
                periodo_summary.columns = ['Período', 'Total Desmatado (km²)', 'Média/Ano (km²)', 'Anos Analisados']
                st.dataframe(periodo_summary, use_container_width=True, hide_index=True) 