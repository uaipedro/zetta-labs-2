"""
Script principal para orquestrar o treinamento de todos os modelos.
Versão modular e organizada do pipeline de machine learning.
"""

import pandas as pd
import json
import warnings
from datetime import datetime
import numpy as np

from .base import (
    setup_directories, load_data, create_walk_forward_splits,
    prepare_features_tabular, prepare_sequences_temporal,
    evaluate_model_performance, save_model_artifact,
    load_existing_results, get_completed_experiments,
    save_checkpoint, RESULTS_DIR, RUN_ID
)

from .xgboost_model import XGBoostTrainer
from .random_forest_model import RandomForestTrainer


warnings.filterwarnings('ignore')


class ModelPipeline:
    """Pipeline principal para treinamento de modelos."""
    
    def __init__(self):
        self.all_results = []
        self.completed_experiments = set()
        self.trainers = {
            'xgboost': XGBoostTrainer(),
            'random_forest': RandomForestTrainer()
        }
    
    def run_pipeline(self):
        """Executa o pipeline completo de modelagem."""
        print("Iniciando pipeline de modelagem modular...")
        print(f"ID da execução: {RUN_ID}")
        print("=" * 60)
        
        setup_directories()
        
        # Carregar resultados existentes
        print("0. Verificando execuções anteriores...")
        existing_results = load_existing_results()
        self.completed_experiments = get_completed_experiments(existing_results)
        self.all_results = existing_results.copy()
        
        if existing_results:
            print(f"Encontrados {len(existing_results)} resultados de execuções anteriores")
            print(f"Experimentos já completados: {len(self.completed_experiments)}")
        
        # Carregar dados
        print("\n1. Carregando dados...")
        df_tabular, _ = load_data()
        
        if df_tabular is None:
            print("Erro: Não foi possível carregar os dados")
            return
        
        # Criar splits walk-forward
        print("\n2. Criando splits walk-forward...")
        splits_tabular = create_walk_forward_splits(df_tabular)
        
        print(f"\n3. Treinando modelos tabulares ({len(splits_tabular)} períodos)...")
        self._train_tabular_models(splits_tabular)
        
        # Salvar resultados finais
        self._save_final_results()
    
    def _train_tabular_models(self, splits):
        """Treina modelos tabulares (XGBoost e Random Forest)."""
        for i, split in enumerate(splits):
            period = i + 1
            test_year = split['test_year']
            
            print(f"\nPeríodo {period}/{len(splits)}: "
                  f"Treino até {max(split['train_years'])}, Teste {test_year}")
            
            # Preparar dados
            X_train, y_train, feature_cols = prepare_features_tabular(split['train_data'])
            X_test, y_test, _ = prepare_features_tabular(split['test_data'])
            
            # Dividir treino em treino/validação para otimização
            val_size = int(0.2 * len(X_train))
            X_val, y_val = X_train.iloc[-val_size:], y_train.iloc[-val_size:]
            X_train_opt, y_train_opt = X_train.iloc[:-val_size], y_train.iloc[:-val_size]
            
            # Treinar modelos tabulares
            for model_type in ['xgboost', 'random_forest']:
                self._train_single_model(
                    model_type, period, test_year,
                    X_train_opt, y_train_opt, X_val, y_val,
                    X_train, y_train, X_test, y_test,
                    feature_cols, split['test_data']
                )
    

    
    def _train_single_model(self, model_type, period, test_year,
                           X_train_opt, y_train_opt, X_val, y_val,
                           X_train, y_train, X_test, y_test, 
                           feature_cols, test_data):
        """Treina um modelo tabular individual."""
        experiment_key = (period, test_year, model_type)
        
        if experiment_key in self.completed_experiments:
            print(f"  {model_type} já treinado para este período - pulando")
            return
        
        print(f"  Treinando {model_type}...")
        
        try:
            trainer = self.trainers[model_type]
            
            # Otimizar hiperparâmetros
            best_params, best_score = trainer.optimize_hyperparameters(
                X_train_opt, y_train_opt, X_val, y_val
            )
            
            # Treinar modelo final
            model = trainer.train_final_model(X_train, y_train, best_params)
            y_pred = trainer.predict(model, X_test)
            
            # Avaliar performance
            metrics = evaluate_model_performance(y_test, y_pred)
            
            # Obter feature importance
            feature_importance_list = trainer.get_feature_importance(model, feature_cols)
            
            # Criar DataFrame de predições
            predictions_df = test_data.loc[y_test.index, ['CD_MUN', 'ano']].copy()
            predictions_df['y_true'] = y_test.values
            predictions_df['y_pred'] = y_pred

            # Salvar artefatos
            save_model_artifact(
                model, model_type, period, test_year, 
                metrics, best_params, 
                feature_importance=feature_importance_list,
                predictions_df=predictions_df
            )
            
            # Armazenar resultado
            result = {
                'period': period,
                'test_year': test_year,
                'model_type': model_type,
                'best_params': best_params,
                'feature_importance': dict(feature_importance_list),
                **metrics
            }
            
            self.all_results.append(result)
            self.completed_experiments.add(experiment_key)
            
            # Salvar checkpoint
            save_checkpoint(self.all_results, "all_results")
            
            print(f"    RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
            
        except Exception as e:
            print(f"    Erro ao treinar {model_type}: {e}")
    

    
    def _save_final_results(self):
        """Salva resultados finais e exibe resumo."""
        print(f"\n5. Salvando resultados finais...")
        
        if not self.all_results:
            print("Nenhum resultado para salvar")
            return
        
        results_df = pd.DataFrame(self.all_results)
        results_path = RESULTS_DIR / f"model_results_{RUN_ID}.csv"
        results_df.to_csv(results_path, index=False)
        
        # Salvar melhores parâmetros
        best_params = {}
        for trainer_name, trainer in self.trainers.items():
            if hasattr(trainer, 'best_params') and trainer.best_params:
                best_params[trainer_name] = trainer.best_params
        
        if best_params:
            params_path = RESULTS_DIR / f"best_parameters_{RUN_ID}.json"
            with open(params_path, 'w') as f:
                json.dump(best_params, f, indent=2, default=lambda x: str(x) if isinstance(x, np.generic) else x)
        
        # Exibir resumo
        self._display_summary(results_df, results_path)
    
    def _display_summary(self, results_df, results_path):
        """Exibe resumo dos resultados."""
        print(f"\n{'='*60}")
        print("RESUMO DOS RESULTADOS")
        print(f"{'='*60}")
        
        if len(results_df) > 0:
            summary = results_df.groupby('model_type').agg({
                'rmse': ['mean', 'std'],
                'mae': ['mean', 'std'],
                'r2': ['mean', 'std']
            }).round(4)
            
            print(summary)
            
            print(f"\nTotal de experimentos completados: {len(results_df)}")
            print(f"Modelos únicos treinados: {results_df['model_type'].nunique()}")
            print(f"Períodos cobertos: {results_df['period'].nunique()}")
            
            # Melhor modelo por métrica
            best_rmse = results_df.loc[results_df['rmse'].idxmin()]
            best_r2 = results_df.loc[results_df['r2'].idxmax()]
            
            print(f"\nMelhor RMSE: {best_rmse['model_type']} (período {best_rmse['period']}) : {best_rmse['rmse']:.4f}")
            print(f"Melhor R²: {best_r2['model_type']} (período {best_r2['period']}) : {best_r2['r2']:.4f}")
        
        print(f"\nResultados salvos em: {results_path}")
        print("Pipeline de modelagem concluído!")


def main():
    """Função principal para executar o pipeline."""
    pipeline = ModelPipeline()
    pipeline.run_pipeline()


if __name__ == "__main__":
    main() 