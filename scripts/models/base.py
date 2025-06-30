"""
Módulo base com funcionalidades comuns para todos os modelos.
Contém setup de diretórios, carregamento de dados, splits temporais, etc.
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Constantes ---
PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "models"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "results"
CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "checkpoints"

# Configurações para validação walk-forward
MIN_TRAIN_YEARS = 5  # Mínimo de anos para treinar
FORECAST_HORIZON = 1  # Prever 1 ano à frente

# Configurações Optuna
N_TRIALS = 20  # Número de trials para otimização
TIMEOUT = 600  # Timeout em segundos (10 minutos)

# Identificador único para esta execução
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')


def setup_directories():
    """Cria diretórios necessários para salvar modelos e resultados."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def save_checkpoint(data, checkpoint_name):
    """Salva checkpoint intermediário."""
    checkpoint_path = CHECKPOINTS_DIR / f"{checkpoint_name}_{RUN_ID}.pkl"
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Checkpoint salvo: {checkpoint_path}")


def load_checkpoint(checkpoint_name, run_id=None):
    """Carrega checkpoint se existir."""
    if run_id:
        checkpoint_path = CHECKPOINTS_DIR / f"{checkpoint_name}_{run_id}.pkl"
    else:
        # Procurar o checkpoint mais recente
        pattern = f"{checkpoint_name}_*.pkl"
        checkpoints = list(CHECKPOINTS_DIR.glob(pattern))
        if not checkpoints:
            return None
        checkpoint_path = max(checkpoints, key=os.path.getctime)
    
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Checkpoint carregado: {checkpoint_path}")
            return data
        except Exception as e:
            print(f"Erro ao carregar checkpoint {checkpoint_path}: {e}")
            return None
    return None


def save_model_artifact(model, model_type, period, test_year, metrics, params, 
                        feature_importance=None, predictions_df=None, scaler=None):
    """Salva artefatos do modelo individual."""
    artifact_dir = MODELS_DIR / f"period_{period:02d}_{test_year}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    # Salvar modelo
    if model_type in ['xgboost', 'random_forest']:
        model_path = artifact_dir / f"{model_type}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    else:  # LSTM/GRU
        model_path = artifact_dir / f"{model_type}_model.h5"
        model.save(model_path)
        
        # Salvar scaler para modelos temporais
        if scaler is not None:
            scaler_path = artifact_dir / f"{model_type}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
    
    # Salvar predições, se disponíveis
    if predictions_df is not None:
        predictions_path = artifact_dir / f"{model_type}_predictions.parquet"
        predictions_df.to_parquet(predictions_path, index=False)
    
    # Salvar metadados
    metadata = {
        'model_type': model_type,
        'period': period,
        'test_year': test_year,
        'metrics': metrics,
        'best_params': params,
        'timestamp': datetime.now().isoformat()
    }
    
    if feature_importance:
        metadata['feature_importance'] = feature_importance
    
    metadata_path = artifact_dir / f"{model_type}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else o)

    
    print(f"Artefatos salvos para {model_type} período {period}: {artifact_dir}")
    return artifact_dir


def load_existing_results():
    """Carrega resultados já existentes para continuar de onde parou."""
    # Tentar carregar checkpoint de resultados
    existing_results = load_checkpoint("all_results")
    if existing_results:
        print(f"Carregados {len(existing_results)} resultados existentes")
        return existing_results
    
    # Se não houver checkpoint, tentar reconstruir dos artefatos individuais
    results = []
    if MODELS_DIR.exists():
        for period_dir in MODELS_DIR.iterdir():
            if period_dir.is_dir() and period_dir.name.startswith('period_'):
                for metadata_file in period_dir.glob('*_metadata.json'):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        result = {
                            'period': metadata['period'],
                            'test_year': metadata['test_year'],
                            'model_type': metadata['model_type'],
                            'best_params': metadata['best_params'],
                            **metadata['metrics']
                        }
                        results.append(result)
                    except Exception as e:
                        print(f"Erro ao carregar metadata {metadata_file}: {e}")
    
    if results:
        print(f"Reconstruídos {len(results)} resultados de artefatos existentes")
    
    return results


def get_completed_experiments(existing_results):
    """Retorna set de experimentos já completados."""
    completed = set()
    for result in existing_results:
        key = (result['period'], result['test_year'], result['model_type'])
        completed.add(key)
    return completed


def load_data():
    """Carrega os dados processados."""
    try:
        tabular_path = PROCESSED_DIR / "analytical_table_tabular.parquet"
        temporal_path = PROCESSED_DIR / "analytical_table_temporal.parquet"
        
        df_tabular = pd.read_parquet(tabular_path)
        df_temporal = pd.read_parquet(temporal_path)
        
        print(f"Dados tabulares: {df_tabular.shape}")
        print(f"Dados temporais: {df_temporal.shape}")
        
        return df_tabular, df_temporal
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None, None


def create_walk_forward_splits(df, target_col='desmatamento_km2'):
    """
    Cria splits para validação walk-forward.
    Tomada de decisão: Usamos walk-forward para simular cenário real onde 
    sempre predizemos o futuro baseado no passado.
    """
    anos_unicos = sorted(df['ano'].unique())
    splits = []
    
    for i in range(MIN_TRAIN_YEARS, len(anos_unicos)):
        train_years = anos_unicos[:i]
        test_year = anos_unicos[i]
        
        train_data = df[df['ano'].isin(train_years)]
        test_data = df[df['ano'] == test_year]
        
        if len(train_data) > 0 and len(test_data) > 0:
            splits.append({
                'train_years': train_years,
                'test_year': test_year,
                'train_data': train_data,
                'test_data': test_data
            })
    
    print(f"Criados {len(splits)} splits walk-forward")
    return splits


def prepare_features_tabular(df, target_col='desmatamento_km2'):
    """Prepara features para modelos tabulares."""
    # Excluir colunas identificadoras e o target
    exclude_cols = ['CD_MUN', 'NM_MUN', 'ano', target_col, 'periodo_presidencial', 'bioma']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Tratar valores infinitos e NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    return X, y, feature_cols


def prepare_sequences_temporal(df, target_col='desmatamento_km2', seq_length=3):
    """
    Prepara sequências temporais para LSTM/GRU.
    Tomada de decisão: Usamos sequências de 3 anos para capturar tendências.
    """
    df_sorted = df.sort_values(['CD_MUN', 'ano']).copy()
    
    # Features para modelos temporais (CORREÇÃO: remover desmatamento_km2 das features para evitar data leakage)
    feature_cols = ['ibama_total_autos', 'ibama_autos_flora', 
                   'ibama_valor_total_multas', 'ano_normalizado', 'ano_transicao']
    
    # Verificar quais colunas estão disponíveis e adicionar se existirem
    available_cols = df_sorted.columns.tolist()
    optional_cols = ['periodo_presidencial_encoded', 'desmatamento_lag1', 'desmatamento_ma3', 'desmatamento_crescimento']
    
    for col in optional_cols:
        if col in available_cols:
            feature_cols.append(col)
    
    X_sequences = []
    y_sequences = []
    metadata = []
    
    for municipio in df_sorted['CD_MUN'].unique():
        # CORREÇÃO: Adicionar 'ano' para podermos separar as sequências de treino/teste depois
        mun_data = df_sorted[df_sorted['CD_MUN'] == municipio][feature_cols + [target_col, 'ano']].copy()
        
        # CORREÇÃO: Remover NaNs introduzidos por features de lag/rolling
        # Isso garante que apenas sequências completas sejam criadas
        mun_data.dropna(inplace=True)
        
        if len(mun_data) >= seq_length + 1:
            for i in range(len(mun_data) - seq_length):
                X_seq = mun_data.iloc[i:i+seq_length][feature_cols].values
                y_val = mun_data.iloc[i+seq_length][target_col]
                # CORREÇÃO: Capturar o ano do target para separar treino/teste
                target_ano = mun_data.iloc[i+seq_length]['ano']
                
                X_sequences.append(X_seq)
                y_sequences.append(y_val)
                metadata.append({
                    'municipio': municipio,
                    'target_year': target_ano,
                })
    
    return np.array(X_sequences), np.array(y_sequences), metadata, feature_cols


def evaluate_model_performance(y_true, y_pred):
    """Calcula métricas de performance."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


class BaseModelTrainer:
    """Classe base para treinadores de modelos."""
    
    def __init__(self, model_type):
        self.model_type = model_type
        self.best_params = {}
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val):
        """Método abstrato para otimização de hiperparâmetros."""
        raise NotImplementedError("Subclasses devem implementar este método")
    
    def train_final_model(self, X_train, y_train, best_params):
        """Método abstrato para treinar modelo final."""
        raise NotImplementedError("Subclasses devem implementar este método")
    
    def predict(self, model, X_test):
        """Método abstrato para fazer predições."""
        raise NotImplementedError("Subclasses devem implementar este método") 