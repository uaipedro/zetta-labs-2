"""
Módulo responsável pelo carregamento e cache de dados
"""
import streamlit as st
import pandas as pd
import geopandas as gpd
import pickle
import json
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Importar configurações usando path absoluto
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from config.settings import (
    PROCESSED_DIR, MODELS_DIR, CHECKPOINTS_DIR, 
    DATA_DIR, CACHE_TTL
)


class DataLoader:
    """Classe para gerenciar carregamento de dados com cache otimizado"""
    
    @staticmethod
    @st.cache_data(ttl=CACHE_TTL)
    def load_base_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Carrega dados base do projeto"""
        try:
            df_base = pd.read_parquet(PROCESSED_DIR / "analytical_base_table.parquet")
            
            # Dados opcionais
            model_comparison = None
            predictions = None
            
            try:
                model_comparison = pd.read_parquet(PROCESSED_DIR / "model_comparison_data.parquet")
            except:
                pass
                
            try:
                predictions = pd.read_parquet(PROCESSED_DIR / "predictions_analysis_sample.parquet")
            except:
                pass
                
            return df_base, model_comparison, predictions
        except Exception as e:
            st.error(f"Erro ao carregar dados: {e}")
            return None, None, None
    
    @staticmethod
    @st.cache_data(ttl=CACHE_TTL)
    def load_model_results() -> Optional[pd.DataFrame]:
        """Carrega resultados dos modelos treinados"""
        try:
            # Tentar checkpoint mais recente
            checkpoints = list(CHECKPOINTS_DIR.glob("all_results_*.pkl"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                with open(latest_checkpoint, 'rb') as f:
                    results = pickle.load(f)
                return pd.DataFrame(results)
            
            # Fallback para parquet
            comparison_file = PROCESSED_DIR / "model_comparison_data.parquet"
            if comparison_file.exists():
                return pd.read_parquet(comparison_file)
            
            return None
        except Exception as e:
            st.warning(f"Erro ao carregar resultados dos modelos: {e}")
            return None
    
    @staticmethod
    @st.cache_data(ttl=CACHE_TTL)
    def load_best_model() -> Tuple[Optional[Any], Optional[Dict], Optional[pd.Series]]:
        """Carrega o melhor modelo com metadados"""
        try:
            model_results = DataLoader.load_model_results()
            if model_results is None:
                return None, None, None
            
            # Encontrar melhor modelo XGBoost
            xgb_results = model_results[model_results['model_type'] == 'xgboost']
            if xgb_results.empty:
                return None, None, None
                
            best_result = xgb_results.loc[xgb_results['r2'].idxmax()]
            period = int(best_result['period'])
            test_year = best_result['test_year']
            
            # Carregar modelo e metadados
            model_dir = MODELS_DIR / f"period_{period:02d}_{test_year}"
            model_file = model_dir / "xgboost_model.pkl"
            metadata_file = model_dir / "xgboost_metadata.json"
            
            model = None
            metadata = None
            
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            return model, metadata, best_result
            
        except Exception as e:
            st.warning(f"Erro ao carregar melhor modelo: {e}")
            return None, None, None
    
    @staticmethod
    @st.cache_data(ttl=CACHE_TTL)
    def load_feature_importance() -> Optional[pd.DataFrame]:
        """Carrega dados de importância das features"""
        try:
            model_results = DataLoader.load_model_results()
            if model_results is None:
                return None
                
            feature_data = []
            
            for _, result in model_results.iterrows():
                period = int(result['period'])
                test_year = result['test_year']
                model_type = result['model_type']
                
                model_dir = MODELS_DIR / f"period_{period:02d}_{test_year}"
                metadata_file = model_dir / f"{model_type}_metadata.json"
                
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    if 'feature_importance' in metadata:
                        feat_imp = metadata['feature_importance']
                        
                        if isinstance(feat_imp, list):
                            feat_imp = {item[0]: item[1] for item in feat_imp}
                        
                        for feature, importance in feat_imp.items():
                            feature_data.append({
                                'model_type': model_type,
                                'period': period,
                                'test_year': test_year,
                                'feature': feature,
                                'importance': importance
                            })
            
            return pd.DataFrame(feature_data) if feature_data else None
            
        except Exception as e:
            st.warning(f"Erro ao carregar feature importance: {e}")
            return None
    
    @staticmethod
    @st.cache_data(ttl=3600)  # Cache por 1 hora
    def load_shapefile() -> Optional[gpd.GeoDataFrame]:
        """Carrega shapefile dos municípios do Pará"""
        try:
            shapefile_path = DATA_DIR / "PA_Municipios_2024/PA_Municipios_2024.shp"
            gdf = gpd.read_file(shapefile_path)

            # Reprojetar para um sistema de coordenadas em metros (SIRGAS 2000 / UTM zone 22S)
            # Isso é crucial para cálculos de área precisos.
            gdf = gdf.to_crs("EPSG:31982")

            if 'CD_MUN' in gdf.columns:
                # Garante que CD_MUN seja Int64 para consistência com outros dataframes
                gdf['CD_MUN'] = pd.to_numeric(gdf['CD_MUN'], errors='coerce').astype('Int64')
            return gdf
        except Exception as e:
            st.error(f"Erro ao carregar shapefile: {e}")
            return None
    
    @staticmethod
    @st.cache_data(ttl=CACHE_TTL)
    def load_executive_summary() -> Optional[pd.DataFrame]:
        """Carrega sumário executivo se disponível"""
        try:
            summary_file = PROCESSED_DIR / "executive_summary.parquet"
            if summary_file.exists():
                return pd.read_parquet(summary_file)
            return None
        except:
            return None
    
    @staticmethod
    @st.cache_data(ttl=CACHE_TTL)
    def load_critical_years() -> Optional[pd.DataFrame]:
        """Carrega dados de anos críticos"""
        try:
            critical_file = PROCESSED_DIR / "critical_years_data.parquet"
            if critical_file.exists():
                return pd.read_parquet(critical_file)
            return None
        except:
            return None 