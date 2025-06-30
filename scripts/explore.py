import pandas as pd
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# --- Constantes ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

IPS_PATH = DATA_DIR / "ips_2024.csv"
DESMATAMENTO_PATH = DATA_DIR / "desmatamento_anual_municipios.parquet"
IBAMA_AUTOS_DIR = DATA_DIR / "ibama" / "autos_infracao_csv"


# --- Funções de Carregamento (Simplificadas para esta etapa) ---

def load_ips_data():
    """Carrega os dados do Índice de Progresso Social (IPS)."""
    try:
        # Tomada de decisão: Corrigindo o separador para vírgula com base na análise do erro.
        # O padrão para arquivos CSV com separador de vírgula é usar o ponto como decimal.
        return pd.read_csv(IPS_PATH, sep=',', decimal='.')
    except Exception as e:
        print(f"Não foi possível carregar os dados do IPS: {e}")
        return None

def load_desmatamento_data():
    """Carrega os dados de desmatamento."""
    try:
        return pd.read_parquet(DESMATAMENTO_PATH)
    except Exception as e:
        print(f"Não foi possível carregar os dados de desmatamento: {e}")
        return None

def load_ibama_data():
    """Carrega e concatena os dados de autos de infração do IBAMA."""
    csv_files = glob.glob(str(IBAMA_AUTOS_DIR / "auto_infracao_ano_*.csv"))
    if not csv_files:
        print(f"Nenhum arquivo do IBAMA encontrado em {IBAMA_AUTOS_DIR}")
        return None
    
    dfs = [pd.read_csv(f, sep=';', decimal=',', encoding='latin1', low_memory=False) for f in sorted(csv_files)]
    return pd.concat(dfs, ignore_index=True)


# --- Funções de Análise e Visualização ---

def setup_figures_dir():
    """Cria o diretório de saída para os gráficos, se não existir."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def plot_missing_values(df: pd.DataFrame, name: str):
    """
    Cria e salva um gráfico de barras com a porcentagem de valores ausentes por coluna.
    """
    # Tomada de decisão: Adicionar verificação para DFs vazios para evitar erro de divisão por zero.
    if df.empty:
        print(f"[{name}] DataFrame está vazio. Nenhum gráfico de valores ausentes gerado.")
        return

    missing_percentage = df.isnull().sum() / len(df) * 100
    missing_percentage = missing_percentage[missing_percentage > 0].sort_values(ascending=False)

    if missing_percentage.empty:
        print(f"[{name}] Nenhum valor ausente encontrado.")
        return

    plt.figure(figsize=(12, 8))
    sns.barplot(x=missing_percentage.index, y=missing_percentage.values)
    plt.xticks(rotation=90)
    plt.title(f'Porcentagem de Valores Ausentes - {name}')
    plt.ylabel('Porcentagem (%)')
    plt.xlabel('Colunas')
    plt.tight_layout()
    
    figure_path = FIGURES_DIR / f"missing_values_{name}.png"
    plt.savefig(figure_path)
    plt.close()
    print(f"[{name}] Gráfico de valores ausentes salvo em: {figure_path}")


def main():
    """
    Função principal para executar a etapa de Exploração (Explore) do projeto.
    Filtra os dados para o estado do Pará e analisa a qualidade dos dados (valores ausentes).
    """
    print("Iniciando a etapa de Explore (Exploração)...")
    setup_figures_dir()

    # Carregar dados
    df_ips = load_ips_data()
    df_desmatamento = load_desmatamento_data()
    df_ibama = load_ibama_data()

    # Filtrar para o Pará (PA)
    # Tomada de decisão: É crucial garantir que a filtragem seja consistente.
    # Verificamos os nomes das colunas de estado ('UF' ou 'ESTADO') em cada tabela
    # para aplicar o filtro corretamente.
    if df_ips is not None:
        print(f"\n[IPS] Colunas disponíveis: {df_ips.columns.to_list()}")
        df_ips_pa = df_ips[df_ips['UF'] == 'PA'].copy()
        print(f"\n[IPS] Registros totais: {len(df_ips)}. Registros para o Pará: {len(df_ips_pa)}")
        if not df_ips_pa.empty:
            plot_missing_values(df_ips_pa, "ips_pa")

    if df_desmatamento is not None:
        df_desmatamento_pa = df_desmatamento[df_desmatamento['ESTADO'] == 'PA'].copy()
        print(f"\n[Desmatamento] Registros totais: {len(df_desmatamento)}. Registros para o Pará: {len(df_desmatamento_pa)}")
        if not df_desmatamento_pa.empty:
            plot_missing_values(df_desmatamento_pa, "desmatamento_pa")

    if df_ibama is not None:
        df_ibama_pa = df_ibama[df_ibama['UF'] == 'PA'].copy()
        print(f"\n[IBAMA] Registros totais: {len(df_ibama)}. Registros para o Pará: {len(df_ibama_pa)}")
        if not df_ibama_pa.empty:
            plot_missing_values(df_ibama_pa, "ibama_pa")
    
    print("\nEtapa de Explore concluída.")
    print(f"Verifique os gráficos no diretório: {FIGURES_DIR}")

if __name__ == "__main__":
    main() 