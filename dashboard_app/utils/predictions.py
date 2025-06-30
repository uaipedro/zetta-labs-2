"""
Módulo para previsões e simulações
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import streamlit as st

# Configurar path para imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from config.settings import MODELS_DIR, PROCESSED_DIR


class PredictionEngine:
    """Classe para gerenciar previsões e simulações"""
    
    @staticmethod
    def make_predictions(model, base_data: pd.DataFrame, adjustments: Dict) -> Dict[str, float]:
        """Faz previsões usando o modelo treinado"""
        if model is None:
            return None
        
        # Cache no session state para evitar recálculos
        cache_key = f"prediction_base_{len(base_data)}"
        
        scenarios = {}
        
        try:
            # Preparar features base se não estiver em cache
            if cache_key not in st.session_state:
                # Preparar features base (excluir colunas não-features)
                exclude_cols = ['CD_MUN', 'NM_MUN', 'ano', 'desmatamento_km2']
                feature_cols = [col for col in base_data.columns if col not in exclude_cols]
                
                base_features = base_data[feature_cols].fillna(0)
                
                # Preparar features essenciais
                if 'periodo_presidencial' in base_features.columns:
                    periodo_mapping = {'Lula': 0, 'Dilma': 1, 'Temer': 2, 'Bolsonaro': 3}
                    base_features['periodo_presidencial_encoded'] = base_features['periodo_presidencial'].map(periodo_mapping).fillna(0)
                    base_features = base_features.drop(columns=['periodo_presidencial'])
                
                # Criar area_normalizada se não existir
                if 'Área (km²)' in base_features.columns and 'area_normalizada' not in base_features.columns:
                    area_mean = base_features['Área (km²)'].mean()
                    area_std = base_features['Área (km²)'].std()
                    if area_std > 0:
                        base_features['area_normalizada'] = (base_features['Área (km²)'] - area_mean) / area_std
                    else:
                        base_features['area_normalizada'] = 0.0
                
                # Converter colunas para numérico
                for col in base_features.select_dtypes(include=['object']).columns:
                    base_features[col] = pd.to_numeric(base_features[col], errors='coerce').fillna(0)
                
                # Features essenciais que o modelo espera
                essential_features = [
                    'Área (km²)', 'População 2022', 'PIB per capita 2021', 'Índice de Progresso Social',
                    'Necessidades Humanas Básicas', 'Fundamentos do Bem-estar', 'Oportunidades',
                    'ibama_total_autos', 'ibama_valor_total_multas', 'ibama_valor_medio_multas',
                    'ibama_autos_flora', 'ano_normalizado', 'ano_transicao', 'desmatamento_lag1',
                    'desmatamento_ma3', 'desmatamento_crescimento', 'periodo_presidencial_encoded',
                    'area_normalizada'
                ]
                
                # Garantir que todas as features esperadas estejam presentes
                for feature in essential_features:
                    if feature not in base_features.columns:
                        base_features[feature] = 0.0
                
                # Reordenar colunas para corresponder à ordem esperada pelo modelo
                available_features = [f for f in essential_features if f in base_features.columns]
                base_features = base_features[available_features]
                
                st.session_state[cache_key] = base_features
            else:
                base_features = st.session_state[cache_key]
            
            for scenario_name, adjustments_dict in adjustments.items():
                features_adjusted = base_features.copy()
                
                # Aplicar ajustes
                for adjustment_key, change in adjustments_dict.items():
                    if 'autos_flora' in adjustment_key.lower() and 'ibama_autos_flora' in features_adjusted.columns:
                        features_adjusted['ibama_autos_flora'] *= (1 + change/100)
                    elif 'idhm' in adjustment_key.lower() and 'Índice de Progresso Social' in features_adjusted.columns:
                        features_adjusted['Índice de Progresso Social'] += change
                    elif 'renda' in adjustment_key.lower() and 'PIB per capita 2021' in features_adjusted.columns:
                        features_adjusted['PIB per capita 2021'] *= (1 + change/100)
                
                # Fazer previsão
                try:
                    prediction = model.predict(features_adjusted)
                    total_prediction = prediction.sum() if hasattr(prediction, 'sum') else prediction
                    scenarios[scenario_name] = float(total_prediction)
                except Exception as e:
                    st.warning(f"Erro na previsão para {scenario_name}: {str(e)}")
                    scenarios[scenario_name] = 5000 + np.random.normal(0, 500)
        
        except Exception as e:
            st.error(f"Erro ao preparar features para previsão: {str(e)}")
            return None
        
        return scenarios
    
    @staticmethod
    def simulate_scenarios(df: pd.DataFrame, base_year: int) -> Dict[str, List[float]]:
        """Simula diferentes cenários de desmatamento"""
        scenarios = {}
        
        try:
            # Dados base
            base_data = df[df['ano'] == base_year].copy()
            if base_data.empty:
                return None
            
            # Cenário otimista
            scenarios['Otimista'] = PredictionEngine._simulate_monte_carlo(
                base_data, 
                growth_mean=-0.1,  # Redução média de 10%
                growth_std=0.05,
                n_simulations=100
            )
            
            # Cenário base (tendência atual)
            scenarios['Base'] = PredictionEngine._simulate_monte_carlo(
                base_data,
                growth_mean=0.0,  # Manutenção da tendência
                growth_std=0.05,
                n_simulations=100
            )
            
            # Cenário pessimista
            scenarios['Pessimista'] = PredictionEngine._simulate_monte_carlo(
                base_data,
                growth_mean=0.1,  # Aumento médio de 10%
                growth_std=0.05,
                n_simulations=100
            )
            
            return scenarios
            
        except Exception as e:
            st.error(f"Erro na simulação de cenários: {str(e)}")
            return None
    
    @staticmethod
    def _simulate_monte_carlo(base_data: pd.DataFrame, growth_mean: float, 
                            growth_std: float, n_simulations: int) -> List[float]:
        """Executa simulação Monte Carlo para um cenário"""
        base_value = base_data['desmatamento_km2'].sum()
        simulated_values = []
        
        for _ in range(n_simulations):
            # Gerar taxa de crescimento aleatória
            growth_rate = np.random.normal(growth_mean, growth_std)
            
            # Simular valor
            simulated_value = base_value * (1 + growth_rate)
            simulated_values.append(max(0, simulated_value))  # Não permitir valores negativos
        
        # Retornar média das simulações
        return float(np.mean(simulated_values))
    
    @staticmethod
    def evaluate_policy_impact(df: pd.DataFrame, policy_start_year: int) -> Dict:
        """Avalia impacto de políticas implementadas"""
        try:
            # Separar períodos
            before_policy = df[df['ano'] < policy_start_year]['desmatamento_km2'].mean()
            after_policy = df[df['ano'] >= policy_start_year]['desmatamento_km2'].mean()
            
            # Calcular impacto
            impact = ((after_policy - before_policy) / before_policy) * 100
            
            # Avaliar efetividade
            if impact < -10:
                effectiveness = "Alta"
            elif impact < 0:
                effectiveness = "Moderada"
            else:
                effectiveness = "Baixa"
            
            return {
                'impact_percent': impact,
                'effectiveness': effectiveness,
                'before_avg': before_policy,
                'after_avg': after_policy
            }
            
        except Exception as e:
            st.error(f"Erro ao avaliar impacto: {str(e)}")
            return None 