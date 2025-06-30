"""
Script de modelagem refatorado - versão modular.
Este script agora usa a estrutura modular organizada em scripts separados.
"""

import warnings
import sys
import os
from pathlib import Path

# Adicionar o diretório scripts ao path para importações relativas
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from models.trainer import main as run_pipeline

warnings.filterwarnings('ignore')


def main():
    """
    Função principal que executa o pipeline de modelagem usando a estrutura modular.
    
    A refatoração organiza o código em:
    - models/base.py: Funcionalidades comuns
    - models/xgboost_model.py: Modelo XGBoost específico
    - models/random_forest_model.py: Modelo Random Forest específico  
    - models/lstm_model.py: Modelo LSTM específico
    - models/gru_model.py: Modelo GRU específico
    - models/trainer.py: Orquestrador principal
    """
    print("🔧 Executando pipeline de modelagem refatorado...")
    print("📁 Estrutura modular:")
    print("   ├── models/base.py - Funcionalidades comuns")
    print("   ├── models/xgboost_model.py - XGBoost")
    print("   ├── models/random_forest_model.py - Random Forest")
    print("   └── models/trainer.py - Orquestrador principal")
    print("   └── scripts/assess.py - Análise de resultados")
    print("")
    
    # Executar pipeline modular
    run_pipeline()








if __name__ == "__main__":
    main() 