"""
Componente de Análise Espacial
"""
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import sys
from pathlib import Path
from typing import Dict, Optional

# Configurar path para imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from utils.metrics import MetricsCalculator
from utils.visualizations import Visualizer
from config.settings import MAP_HEIGHT


class SpatialAnalyzer:
    """Classe para análise espacial e criação de mapas"""
    pass


def render_spatial_analysis(df: pd.DataFrame, gdf: gpd.GeoDataFrame) -> None:
    """Renderiza a análise espacial do dashboard"""
    
    st.markdown("### 🗺️ Análise Espacial")
    
    # Seletor de ano como slider para ocupar a largura total
    selected_year = st.slider(
        "📅 Selecione o Ano:",
        min_value=2008,
        max_value=2023,
        value=2023,
        key="spatial_year_slider"
    )
    
    # Layout lado a lado
    col_left, col_right = st.columns([1, 1])
    
    # LADO ESQUERDO: Indicadores Contextuais (unificados)
    with col_left:
        st.subheader("🏛️ Indicadores Contextuais")
        
        # Agrupar indicadores sociais e de fiscalização em uma única lista
        indicator_options = [
            {'name': 'Índice de Progresso Social', 'column': 'Índice de Progresso Social', 'color': 'Greens', 'inverse': True},
            {'name': 'Necessidades Básicas', 'column': 'Necessidades Humanas Básicas', 'color': 'Greens', 'inverse': True},
            {'name': 'Bem-estar', 'column': 'Fundamentos do Bem-estar', 'color': 'Greens', 'inverse': True},
            {'name': 'Oportunidades', 'column': 'Oportunidades', 'color': 'Greens', 'inverse': True},
            {'name': 'PIB per capita', 'column': 'PIB per capita 2021', 'color': 'Blues', 'inverse': False},
            {'name': 'População', 'column': 'População 2022', 'color': 'Oranges', 'inverse': False},
            {'name': 'Total de Autuações (IBAMA)', 'column': 'ibama_total_autos', 'color': 'Reds', 'inverse': False},
            {'name': 'Autuações de Flora (IBAMA)', 'column': 'ibama_autos_flora', 'color': 'Reds', 'inverse': False}
        ]
        
        # Seletor de indicador único
        selected_indicator = st.selectbox(
            "Indicador:",
            [opt['name'] for opt in indicator_options],
            key="indicator_name"
        )
        
        # Encontrar configuração do indicador selecionado
        indicator_config = next(
            opt for opt in indicator_options 
            if opt['name'] == selected_indicator
        )
        
        # Criar mapa do indicador
        year_data = df[df['ano'] == selected_year].copy()
        
        if not year_data.empty and gdf is not None:
            # Agregar dados por município
            municipal_data = year_data.groupby(['NM_MUN', 'CD_MUN']).agg({
                indicator_config['column']: 'mean'
            }).reset_index()
            
            # Criar mapa
            fig_indicator = Visualizer.create_choropleth_map(
                gdf=gdf,
                data=municipal_data,
                value_column=indicator_config['column'],
                name_column='NM_MUN',
                code_column='CD_MUN',
                title=f"{indicator_config['name']} - {selected_year}",
                color_scale=indicator_config['color'],
                inverse_scale=indicator_config['inverse']
            )
            
            if fig_indicator:
                st.plotly_chart(fig_indicator, use_container_width=True)
                
                # Estatísticas do indicador com nome do município
                stats_data = municipal_data.dropna(subset=[indicator_config['column']])
                if not stats_data.empty:
                    min_row = stats_data.loc[stats_data[indicator_config['column']].idxmin()]
                    max_row = stats_data.loc[stats_data[indicator_config['column']].idxmax()]
                    mean_val = stats_data[indicator_config['column']].mean()

                    st.markdown(f"**Média:** {mean_val:.2f}")
                    st.markdown(f"**Mínimo:** {min_row[indicator_config['column']]:.2f} (*{min_row['NM_MUN']}*)")
                    st.markdown(f"**Máximo:** {max_row[indicator_config['column']]:.2f} (*{max_row['NM_MUN']}*)")

            else:
                st.error("Erro ao criar mapa do indicador")
        else:
            st.warning("Sem dados disponíveis para o ano selecionado")
    
    # LADO DIREITO: Desmatamento
    with col_right:
        st.subheader("🌳 Desmatamento")
        
        # Modo de visualização com selectbox
        view_mode = st.selectbox(
            "Modo de Visualização:",
            options=['percentual', 'absoluto'],
            format_func=lambda x: '📊 Percentual da área' if x == 'percentual' else '🗺️ Área em km²',
            key="deforestation_view_mode"
        )
        
        # Criar mapa de desmatamento
        if not year_data.empty and gdf is not None:
            # Agregar dados por município
            deforest_data = year_data.groupby(['NM_MUN', 'CD_MUN']).agg({
                'desmatamento_km2': 'sum'
            }).reset_index()
            
            if view_mode == 'percentual':
                # Calcular área dos municípios
                if 'area_municipio_km2' not in gdf.columns:
                    gdf['area_municipio_km2'] = gdf.geometry.area / 1000000
                
                # O merge agora pode ser feito diretamente com Int64, que é mais robusto
                # Assegurar que os tipos são consistentes
                deforest_data['CD_MUN'] = deforest_data['CD_MUN'].astype('Int64')
                gdf['CD_MUN'] = gdf['CD_MUN'].astype('Int64')

                # Merge com áreas
                try:
                    deforest_data = deforest_data.merge(
                        gdf[['CD_MUN', 'area_municipio_km2']].dropna(subset=['CD_MUN']),
                        on='CD_MUN',
                        how='left'
                    )
                    
                    # Calcular percentual
                    deforest_data['desmatamento_percentual'] = (
                        deforest_data['desmatamento_km2'] / deforest_data['area_municipio_km2'] * 100
                    ).fillna(0).clip(0, 100)
                    
                    value_col = 'desmatamento_percentual'
                    title = f"Desmatamento (% da área) - {selected_year}"
                except Exception as e:
                    st.error(f"Erro ao calcular percentuais: {str(e)}")
                    value_col = 'desmatamento_km2'
                    title = f"Desmatamento (km²) - {selected_year}"
            else:
                value_col = 'desmatamento_km2'
                title = f"Desmatamento (km²) - {selected_year}"
            
            # Criar mapa
            fig_deforest = Visualizer.create_choropleth_map(
                gdf=gdf,
                data=deforest_data,
                value_column=value_col,
                name_column='NM_MUN',
                code_column='CD_MUN',
                title=title,
                color_scale='Reds',
                inverse_scale=False
            )
            
            if fig_deforest:
                st.plotly_chart(fig_deforest, use_container_width=True)
                
                # Estatísticas do desmatamento com nome do município
                stats_data = deforest_data.dropna(subset=[value_col])
                if not stats_data.empty:
                    min_row = stats_data.loc[stats_data[value_col].idxmin()]
                    max_row = stats_data.loc[stats_data[value_col].idxmax()]
                    mean_val = stats_data[value_col].mean()
                    sum_val = stats_data[value_col].sum()
                    
                    unit = "%" if view_mode == 'percentual' else "km²"

                    # Formatar com base na unidade
                    if unit == "%":
                        mean_str = f"{mean_val:.2f}{unit}"
                        min_str = f"{min_row[value_col]:.2f}{unit}"
                        max_str = f"{max_row[value_col]:.2f}{unit}"
                    else:
                        mean_str = f"{mean_val:,.0f} {unit}"
                        min_str = f"{min_row[value_col]:,.0f} {unit}"
                        max_str = f"{max_row[value_col]:,.0f} {unit}"

                    st.markdown(f"**Média:** {mean_str}")
                    st.markdown(f"**Mínimo:** {min_str} (*{min_row['NM_MUN']}*)")
                    st.markdown(f"**Máximo:** {max_str} (*{max_row['NM_MUN']}*)")

                    if view_mode == 'absoluto':
                        st.markdown(f"**Total Desmatado:** {sum_val:,.0f} km²")
            else:
                st.error("Erro ao criar mapa de desmatamento")
        else:
            st.warning("Sem dados disponíveis para o ano selecionado")
    
    # Análise de correlação espacial
    st.markdown("---")
    st.markdown("#### 🔄 Análise de Correlação Espacial")
    
    # Calcular correlações
    if not year_data.empty:
        correlations = MetricsCalculator.calculate_social_impact_correlation(year_data)
        
        if correlations:
            # Criar matriz de correlação
            correlation_data = []
            for indicator, data in correlations.items():
                correlation_data.append({
                    'Indicador': indicator,
                    'Correlação': data['correlation'],
                    'Força': data['strength'],
                    'Direção': data['direction']
                })
            
            corr_df = pd.DataFrame(correlation_data)
            
            # Visualizar correlações
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Gráfico de correlações
                fig_corr = Visualizer.create_correlation_matrix(
                    df=year_data[['desmatamento_km2'] + [opt['column'] for opt in indicator_options]],
                    variables=['desmatamento_km2'] + [opt['column'] for opt in indicator_options],
                    title=f"Matriz de Correlação - {selected_year}"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with col2:
                # Tabela de correlações
                st.dataframe(
                    corr_df,
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.warning("Não foi possível calcular correlações espaciais")
    else:
        st.warning("Sem dados disponíveis para análise de correlação") 