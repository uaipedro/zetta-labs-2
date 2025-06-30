"""
Módulo específico para treinamento e otimização do modelo XGBoost.
"""

import optuna
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from .base import BaseModelTrainer, N_TRIALS, TIMEOUT


class XGBoostTrainer(BaseModelTrainer):
    """Treinador específico para modelos XGBoost."""
    
    def __init__(self):
        super().__init__('xgboost')
    
    def objective_function(self, trial, X_train, y_train, X_val, y_val):
        """Função objetivo para otimização do XGBoost com Optuna."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        return rmse
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val):
        """Otimiza hiperparâmetros usando Optuna."""
        print(f"Otimizando hiperparâmetros para {self.model_type}...")
        
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: self.objective_function(trial, X_train, y_train, X_val, y_val),
            n_trials=N_TRIALS,
            timeout=TIMEOUT,
            show_progress_bar=True
        )
        
        print(f"Melhor RMSE para {self.model_type}: {study.best_value:.4f}")
        self.best_params = study.best_params
        
        return study.best_params, study.best_value
    
    def train_final_model(self, X_train, y_train, best_params):
        """Treina modelo final com os melhores parâmetros."""
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train)
        return model
    
    def predict(self, model, X_test):
        """Faz predições com o modelo treinado."""
        return model.predict(X_test)
    
    def get_feature_importance(self, model, feature_names):
        """Retorna importância das features."""
        importance = model.feature_importances_
        feature_importance = dict(zip(feature_names, importance))
        return sorted(feature_importance.items(), key=lambda x: x[1], reverse=True) 