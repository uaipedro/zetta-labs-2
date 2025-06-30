"""
Módulo para cálculo de métricas e indicadores
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Configurar path para imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))


class MetricsCalculator:
    """Classe para calcular métricas e indicadores do dashboard"""
    
    @staticmethod
    def calculate_key_metrics(df: pd.DataFrame) -> Dict:
        """Calcula métricas principais do desmatamento"""
        
        # Dados agregados por ano
        yearly_data = df.groupby('ano').agg({
            'desmatamento_km2': 'sum',
            'NM_MUN': 'nunique'
        }).reset_index()
        
        # Métricas básicas
        min_year = int(df['ano'].min())
        max_year = int(df['ano'].max())
        total_desmatamento = df['desmatamento_km2'].sum()
        media_anual = yearly_data['desmatamento_km2'].mean()
        
        # Tendência (últimos 5 anos vs 5 anos anteriores)
        anos_ordenados = yearly_data.sort_values('ano')
        if len(anos_ordenados) >= 10:
            primeiros_5 = anos_ordenados.head(len(anos_ordenados) - 5).tail(5)['desmatamento_km2'].mean()
            ultimos_5 = anos_ordenados.tail(5)['desmatamento_km2'].mean()
            tendencia = ((ultimos_5 - primeiros_5) / primeiros_5) * 100 if primeiros_5 > 0 else 0
        else:
            tendencia = 0 # Não há dados suficientes para uma tendência de 5 anos

        # Pico e vale
        pico_ano = yearly_data.loc[yearly_data['desmatamento_km2'].idxmax(), 'ano']
        pico_valor = yearly_data['desmatamento_km2'].max()
        vale_ano = yearly_data.loc[yearly_data['desmatamento_km2'].idxmin(), 'ano']
        vale_valor = yearly_data['desmatamento_km2'].min()
        
        # Destaques do último ano
        latest_year_data = df[df['ano'] == max_year]
        municipal_latest_agg = latest_year_data.groupby('NM_MUN').agg(
            desmatamento_km2=('desmatamento_km2', 'sum'),
            area_km2=('Área (km²)', 'first')
        ).reset_index()

        municipal_latest_agg['desmatamento_percentual'] = (
            (municipal_latest_agg['desmatamento_km2'] / municipal_latest_agg['area_km2']) * 100
        ).fillna(0)

        municipal_latest = municipal_latest_agg.sort_values(by='desmatamento_km2', ascending=False)
        
        columns_to_show = ['NM_MUN', 'desmatamento_km2', 'area_km2', 'desmatamento_percentual']
        top_5_raw = municipal_latest[columns_to_show].head(5)
        bottom_5_raw = municipal_latest[columns_to_show].tail(5)

        rename_map = {
            'NM_MUN': 'Município',
            'desmatamento_km2': 'Desmatado (km²)',
            'area_km2': 'Área Total (km²)',
            'desmatamento_percentual': 'Desmatado (%)'
        }

        top_5 = top_5_raw.rename(columns=rename_map)
        bottom_5 = bottom_5_raw.rename(columns=rename_map)

        latest_year_stats = {
            'year': max_year,
            'top_5': top_5,
            'bottom_5': bottom_5
        }

        return {
            'total_desmatamento': total_desmatamento,
            'media_anual': media_anual,
            'tendencia_percentual': tendencia,
            'pico_ano': int(pico_ano),
            'pico_valor': pico_valor,
            'vale_ano': int(vale_ano),
            'vale_valor': vale_valor,
            'municipios_total': df['NM_MUN'].nunique(),
            'min_year': min_year,
            'max_year': max_year,
            'latest_year_stats': latest_year_stats
        }
    
    @staticmethod
    def calculate_social_impact_correlation(df: pd.DataFrame) -> Dict:
        """Calcula correlações entre indicadores sociais e desmatamento"""
        
        social_indicators = [
            'Índice de Progresso Social',
            'Necessidades Humanas Básicas',
            'Fundamentos do Bem-estar',
            'Oportunidades',
            'PIB per capita 2021'
        ]
        
        correlations = {}
        for indicator in social_indicators:
            if indicator in df.columns:
                corr = df[indicator].corr(df['desmatamento_km2'])
                correlations[indicator] = {
                    'correlation': corr,
                    'strength': MetricsCalculator._interpret_correlation(corr),
                    'direction': 'Positiva' if corr > 0 else 'Negativa'
                }
        
        return correlations
    
    @staticmethod
    def _interpret_correlation(corr: float) -> str:
        """Interpreta força da correlação"""
        abs_corr = abs(corr)
        if abs_corr > 0.7:
            return "Forte"
        elif abs_corr > 0.5:
            return "Moderada"
        elif abs_corr > 0.3:
            return "Fraca"
        else:
            return "Muito fraca" 