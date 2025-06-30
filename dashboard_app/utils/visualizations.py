"""
Módulo de visualizações reutilizáveis
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import streamlit as st

# Importar configurações usando path absoluto
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from config.settings import CHART_HEIGHT, MAP_HEIGHT, COLORSCALES


class Visualizer:
    """Classe com métodos de visualização reutilizáveis"""
    
    @staticmethod
    def create_kpi_card(title: str, value: float, delta: float = None, 
                       prefix: str = "", suffix: str = "", color: str = "primary") -> None:
        """Cria um card KPI estilizado"""
        if delta is not None:
            delta_color = "green" if delta > 0 else "red"
            delta_icon = "📈" if delta > 0 else "📉"
            delta_text = f"{delta_icon} {abs(delta):.1f}%"
        else:
            delta_text = ""
            
        # CSS customizado para o card
        card_style = f"""
        <div style="
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <h4 style="color: #666; margin: 0;">{title}</h4>
            <h2 style="color: #333; margin: 10px 0;">
                {prefix}{value:,.0f}{suffix}
            </h2>
            <p style="color: {'green' if delta and delta < 0 else 'red' if delta and delta > 0 else '#666'}; 
                      margin: 0; font-size: 14px;">
                {delta_text}
            </p>
        </div>
        """
        st.markdown(card_style, unsafe_allow_html=True)
    
    @staticmethod
    def create_trend_chart(df: pd.DataFrame, x_col: str, y_col: str, 
                          title: str = "", 
                          predictions_df: Optional[pd.DataFrame] = None,
                          historical_predictions_df: Optional[pd.DataFrame] = None,
                          y_axis_title: str = "Valor", hover_template: str = '%{y:,.2f}',
                          color_col: Optional[str] = None) -> go.Figure:
        """
        Cria gráfico de tendência temporal com suporte a previsões.
        - `predictions_df`: para previsões futuras (anexadas ao final).
        - `historical_predictions_df`: para previsões passadas (sobrepostas).
        - `color_col`: se fornecido, agrupa os dados e plota uma linha por grupo.
        """
        fig = go.Figure()

        if color_col and color_col in df.columns:
            # Múltiplas linhas baseadas na coluna de cor
            colors = px.colors.qualitative.Plotly
            for i, (name, group) in enumerate(df.groupby(color_col)):
                sorted_group = group.sort_values(x_col)
                fig.add_trace(go.Scatter(
                    x=sorted_group[x_col],
                    y=sorted_group[y_col],
                    mode='lines+markers',
                    name=str(name).replace('_', ' ').title(),
                    line=dict(color=colors[i % len(colors)], width=2.5),
                    marker=dict(size=7),
                    hovertemplate=f'%{{x}}: {hover_template}<extra></extra>'
                ))
        else:
            # Linha de dados reais
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode='lines+markers',
                name='Real',
                line=dict(color='#2E8B57', width=3),
                marker=dict(size=8),
                hovertemplate=f'%{{x}}: {hover_template}<extra></extra>'
            ))
        
            # Linha de previsões históricas (sobrepostas) - Apenas para gráfico de linha única
            if historical_predictions_df is not None and not historical_predictions_df.empty:
                fig.add_trace(go.Scatter(
                    x=historical_predictions_df[x_col],
                    y=historical_predictions_df[y_col],
                    mode='lines+markers',
                    name='Previsão Histórica',
                    line=dict(color='#FFA500', width=2.5, dash='dot'), # Laranja, pontilhada
                    marker=dict(size=7, symbol='circle-open'),
                    hovertemplate=f'%{{x}}: {hover_template} (previsão)<extra></extra>'
                ))
                
            # Linha de previsões futuras (conectadas ao fim) - Apenas para gráfico de linha única
            if predictions_df is not None and not predictions_df.empty:
                last_real_point = df.iloc[-1]
                plot_df = pd.concat([pd.DataFrame([last_real_point]), predictions_df], ignore_index=True)
                
                fig.add_trace(go.Scatter(
                    x=plot_df[x_col],
                    y=plot_df[y_col],
                    mode='lines+markers',
                    name='Previsão Futura',
                    line=dict(color='#FF6B6B', width=3, dash='dash'),
                    marker=dict(size=10, symbol='diamond'),
                    hovertemplate=f'%{{x}}: {hover_template} (previsão)<extra></extra>'
                ))
            
            # Adicionar anotações de máximo e mínimo (apenas nos dados reais de linha única)
            if not df.empty:
                max_val = df.loc[df[y_col].idxmax()]
                min_val = df.loc[df[y_col].idxmin()]
                
                fig.add_annotation(
                    x=max_val[x_col], y=max_val[y_col],
                    text=f"Máximo: {max_val[y_col]:.2f}",
                    showarrow=True, arrowhead=2, bgcolor="red", bordercolor="red", font=dict(color="white")
                )
                
                fig.add_annotation(
                    x=min_val[x_col], y=min_val[y_col],
                    text=f"Mínimo: {min_val[y_col]:.2f}",
                    showarrow=True, arrowhead=2, bgcolor="green", bordercolor="green", font=dict(color="white")
                )
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis_title="Ano",
            yaxis_title=y_axis_title,
            hovermode='x unified',
            template='plotly_white',
            height=CHART_HEIGHT,
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="center", x=0.5
            )
        )
        
        return fig
    
    @staticmethod
    def create_risk_heatmap(hotspots: List[Dict], title: str = "Matriz de Risco Municipal") -> go.Figure:
        """Cria heatmap de risco por município"""
        # Preparar dados para o heatmap
        data = pd.DataFrame(hotspots)
        
        # Criar matriz de risco
        risk_matrix = data.pivot_table(
            values='risk_score',
            index=pd.cut(data['desmatamento_km2'], bins=5),
            columns=pd.cut(data['crescimento'], bins=5),
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=risk_matrix.values,
            x=['Muito Baixo', 'Baixo', 'Médio', 'Alto', 'Muito Alto'],
            y=['Muito Baixo', 'Baixo', 'Médio', 'Alto', 'Muito Alto'],
            colorscale='RdYlGn_r',
            text=np.round(risk_matrix.values, 1),
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False,
            hovertemplate='Desmatamento: %{y}<br>Crescimento: %{x}<br>Score de Risco: %{z:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis_title="Taxa de Crescimento",
            yaxis_title="Volume de Desmatamento",
            height=CHART_HEIGHT,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_correlation_matrix(df: pd.DataFrame, variables: List[str], 
                                title: str = "Matriz de Correlação") -> go.Figure:
        """Cria matriz de correlação interativa"""
        # Calcular correlações
        corr_matrix = df[variables].corr()
        
        # Criar heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='%{x} x %{y}<br>Correlação: %{z:.2f}<extra></extra>'
        ))
        
        # Ajustar layout
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            height=CHART_HEIGHT + 100,
            template='plotly_white',
            xaxis=dict(tickangle=-45)
        )
        
        return fig
    
    @staticmethod
    def create_scenario_comparison(scenarios: Dict[str, float], 
                                 title: str = "Comparação de Cenários") -> go.Figure:
        """Cria gráfico de comparação de cenários"""
        # Preparar dados
        scenario_names = list(scenarios.keys())
        values = list(scenarios.values())
        colors = ['green' if 'Otimista' in name else 'red' if 'Pessimista' in name else 'blue' 
                 for name in scenario_names]
        
        fig = go.Figure(data=[
            go.Bar(
                x=scenario_names,
                y=values,
                text=[f'{v:,.0f}' for v in values],
                textposition='outside',
                marker_color=colors,
                hovertemplate='%{x}<br>Previsão: %{y:,.0f} km²<extra></extra>'
            )
        ])
        
        # Adicionar linha de referência
        baseline = scenarios.get('Cenário Base', np.mean(values))
        fig.add_hline(
            y=baseline, 
            line_dash="dash", 
            line_color="gray",
            annotation_text=f"Baseline: {baseline:,.0f}"
        )
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis_title="Cenário",
            yaxis_title="Desmatamento Previsto (km²)",
            height=CHART_HEIGHT,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_feature_importance_chart(feature_importance: pd.Series, 
                                      title: str = "Importância das Variáveis") -> go.Figure:
        """Cria gráfico de importância das features a partir de uma Series pandas."""
        # Ordenar valores para exibição correta no gráfico de barras horizontal
        avg_importance = feature_importance.sort_values(ascending=True)
        
        # Criar gráfico horizontal
        fig = go.Figure(data=[
            go.Bar(
                x=avg_importance.values,
                y=avg_importance.index,
                orientation='h',
                marker=dict(
                    color=avg_importance.values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importância")
                ),
                text=[f'{v:.3f}' for v in avg_importance.values],
                textposition='outside',
                hovertemplate='%{y}<br>Importância: %{x:.3f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis_title="Importância Média",
            yaxis_title="",
            height=max(CHART_HEIGHT, len(avg_importance) * 30),
            template='plotly_white',
            margin=dict(l=200)  # Espaço para nomes longos
        )
        
        return fig
    
    @staticmethod
    def create_metrics_evolution_chart(data: pd.DataFrame, metrics: List[str], x: str, title: str) -> go.Figure:
        """Cria um gráfico de evolução para múltiplas métricas ao longo do tempo."""
        df_melted = data.melt(id_vars=[x, 'model_type'], value_vars=metrics, var_name='metrica', value_name='valor')
        
        fig = px.line(
            df_melted,
            x=x,
            y='valor',
            color='model_type',
            line_dash='metrica',
            title=title,
            template='plotly_white',
            height=CHART_HEIGHT,
            labels={
                'valor': 'Valor da Métrica',
                x: 'Ano de Teste',
                'metrica': 'Métrica'
            }
        )
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            legend_title_text='Modelos e Métricas'
        )
        return fig
    
    @staticmethod
    def create_error_distribution_chart(data: pd.DataFrame, x: str, y: str, title: str, color: str) -> go.Figure:
        """Cria um boxplot para visualizar a distribuição de um erro ou métrica."""
        fig = px.box(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            points="all",
            template='plotly_white',
            height=CHART_HEIGHT,
            labels={
                y: y.upper(),
                x: x.replace('_', ' ').title()
            }
        )
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            showlegend=False
        )
        return fig
    
    @staticmethod
    def create_gauge_chart(value: float, title: str, min_val: float = 0, 
                          max_val: float = 100, target: float = None) -> go.Figure:
        """Cria gráfico de gauge para exibir uma métrica"""
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title, 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#2E8B57"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [min_val, min_val + (max_val - min_val) * 0.6], 'color': 'lightgray'},
                    {'range': [min_val + (max_val - min_val) * 0.6, max_val], 'color': 'gray'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': target if target is not None else max_val
                }
            },
            number={'suffix': '%', 'font': {'size': 32}} if max_val == 100 else {}
        ))

        fig.update_layout(
            height=CHART_HEIGHT - 50,
            margin=dict(l=20, r=20, t=50, b=20),
            template='plotly_white'
        )

        return fig

    @staticmethod
    def create_choropleth_map(gdf, data, value_column, name_column, code_column, 
                             title, color_scale='Reds', inverse_scale=False) -> go.Figure:
        """Cria mapa choropleth otimizado com legenda e tratamento de nulos."""
        
        # Etapa 1: Merge de dados e geometrias
        data_to_merge = data.copy()
        if name_column in data_to_merge.columns and name_column in gdf.columns:
            data_to_merge = data_to_merge.drop(columns=[name_column])
        
        merged_gdf = gdf.merge(data_to_merge, on=code_column, how='left')
        
        fig = go.Figure()
        
        # Etapa 2: Tratamento de nulos e preparação de dados para coloração
        if value_column in merged_gdf.columns and not merged_gdf[value_column].empty:
            is_imputed = merged_gdf[value_column].isnull()
            merged_gdf['imputed_flag'] = np.where(is_imputed, "*", "")
            
            if is_imputed.any():
                mean_val = merged_gdf[value_column].mean()
                merged_gdf[value_column] = merged_gdf[value_column].fillna(mean_val)
            
            values = merged_gdf[value_column]
            min_val, max_val = values.min(), values.max()
            is_uniform = max_val == min_val
        else:
            merged_gdf[value_column] = 0
            merged_gdf['imputed_flag'] = " (sem dados)"
            min_val, max_val = 0, 0
            is_uniform = True

        # Etapa 3: Desenhar cada polígono
        for _, row in merged_gdf.iterrows():
            if row.geometry is None:
                continue
            
            lons, lats = [], []
            geom_type = row.geometry.geom_type
            
            # Tratar tanto Polygon (uma área) quanto MultiPolygon (múltiplas áreas)
            if geom_type == 'Polygon':
                coords = list(row.geometry.exterior.coords)
                lons.extend([c[0] for c in coords])
                lats.extend([c[1] for c in coords])
            elif geom_type == 'MultiPolygon':
                for poly in row.geometry.geoms:
                    coords = list(poly.exterior.coords)
                    lons.extend([c[0] for c in coords])
                    lats.extend([c[1] for c in coords])
                    lons.append(None)  # Separador para desenhar polígonos múltiplos
                    lats.append(None)
            else:
                continue
            
            # Pular se não houver coordenadas válidas
            if not lons:
                continue
            
            value = row.get(value_column, 0)
            
            # Etapa 3a: Calcular "intensidade de 'bom'" normalizada (0=ruim, 1=bom)
            normalized = (value - min_val) / (max_val - min_val) if not is_uniform else 0.5
            intensity = normalized if inverse_scale else 1.0 - normalized

            # Etapa 3b: Mapear intensidade para uma faixa visual que evita tons extremos
            display_intensity = 0.15 + 0.80 * intensity

            # Etapa 3c: Calcular a cor com base na escala e na intensidade
            if color_scale in ['Reds', 'Oranges']:
                # Para escalas "ruins", alta intensidade (bom) = cor clara
                color_val = int(255 * display_intensity)
            else:
                # Para escalas "boas" (Greens, Blues), alta intensidade (bom) = cor escura
                color_val = int(255 * (1 - display_intensity))

            # Define a cor final
            if color_scale == 'Reds':
                color = f'rgba(255, {color_val}, {color_val}, 0.85)'
            elif color_scale == 'Greens':
                color = f'rgba({color_val}, 255, {color_val}, 0.85)'
            elif color_scale == 'Blues':
                color = f'rgba({color_val}, {color_val}, 255, 0.85)'
            else:  # Oranges
                # Vai de Laranja (ruim, intensidade baixa) para Amarelo (bom, intensidade alta)
                g_val = 165 + int(90 * display_intensity)
                color = f'rgba(255, {g_val}, 0, 0.85)'

            tooltip_text = f"<b>{row[name_column]}</b><br>{value_column.replace('_', ' ').title()}: {value:.2f}{row['imputed_flag']}"
            if row['imputed_flag'] == '*':
                tooltip_text += " (média)"
                
            fig.add_trace(go.Scatter(
                x=lons, y=lats, fill='toself', fillcolor=color,
                line=dict(color='gray', width=0.5), mode='lines',
                name=row[name_column], text=tooltip_text,
                hoverinfo='text', showlegend=False
            ))

        # Etapa 4: Adicionar a legenda (colorbar)
        if not is_uniform:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(
                    colorscale=color_scale, showscale=True, cmin=min_val, cmax=max_val,
                    colorbar=dict(
                        title=dict(
                            text=value_column.replace('_', ' ').title(),
                            side='right'
                        ),
                        thickness=15, len=0.7,
                        yanchor='middle', y=0.5
                    )
                ),
                hoverinfo='none'
            ))

        # Etapa 5: Configurar layout
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            height=MAP_HEIGHT,
            margin={"r":0,"t":40,"l":0,"b":0},
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, scaleanchor="x"),
            plot_bgcolor='#f9f9f9'
        )
        
        return fig

    @staticmethod
    def plot_model_comparison(df: pd.DataFrame, 
                              models: List[str] = ['random_forest', 'xgboost'],
                              metrics: List[str] = ['MAE', 'RMSE', 'R²'],
                              title: str = "Comparativo de Performance dos Modelos") -> go.Figure:
        """Cria um gráfico de comparação de performance entre modelos"""
        # Filtrar dados
        df_filtered = df[df['model_type'].isin(models)]
        
        # Criar gráfico
        fig = go.Figure()
        
        for metric in metrics:
            fig.add_trace(go.Bar(
                x=df_filtered['model_type'],
                y=df_filtered[metric],
                name=metric,
                text=[f'{v:.2f}' for v in df_filtered[metric]],
                textposition='outside'
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis_title="Modelo",
            yaxis_title="Valor da Métrica",
            height=CHART_HEIGHT,
            template='plotly_white',
            barmode='group',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig