"""
Script para executar o Dashboard v2
"""
import sys
import subprocess
from pathlib import Path

def main():
    """Executa o dashboard componentizado"""
    
    # Adicionar o diretório do dashboard ao path
    dashboard_dir = Path(__file__).parent / "dashboard_app"
    sys.path.insert(0, str(dashboard_dir))
    
    # Executar o streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_dir / "app.py"),
        "--server.port", "8501",
        "--server.headless", "true",
        "--server.runOnSave", "true"  # Habilita hot reload ao salvar arquivos
    ])

if __name__ == "__main__":
    main() 