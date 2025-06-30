import pandas as pd
import glob
from pathlib import Path

# Constantes de caminhos
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
IPS_PATH = DATA_DIR / "ips_2024.csv"
DESMATAMENTO_PATH = DATA_DIR / "desmatamento_anual_municipios.parquet"
IBAMA_AUTOS_DIR = DATA_DIR / "ibama" / "autos_infracao_csv"


def setup_reports_dir():
    """Cria o diretório de saída para os relatórios, se não existir."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def summarize_df(df: pd.DataFrame, name: str):
    """
    Gera um resumo de um DataFrame e salva em um arquivo de texto.
    O resumo inclui as primeiras 5 linhas, informações sobre tipos de dados e valores nulos,
    e estatísticas descritivas.
    """
    report_path = REPORTS_DIR / f"summary_{name}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Resumo para a tabela: {name}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Dimensões do DataFrame:\n")
        f.write(f"{df.shape}\n\n")

        f.write("Primeiras 5 linhas:\n")
        f.write(df.head().to_string())
        f.write("\n\n")

        f.write("Informações do DataFrame (tipos de dados e nulos):\n")
        df.info(buf=f)
        f.write("\n\n")

        f.write("Estatísticas Descritivas:\n")
        f.write(df.describe(include='all').to_string())
        f.write("\n\n")
    print(f"Relatório de resumo salvo em: {report_path}")


def load_and_summarize_ips():
    """Carrega e resume os dados do Índice de Progresso Social (IPS)."""
    try:
        
        df = pd.read_csv(IPS_PATH, sep=',', decimal='.')
        summarize_df(df, "ips_municipios")
        return df
    except FileNotFoundError:
        print(f"Erro: Arquivo {IPS_PATH} não encontrado.")
        return None
    


def load_and_summarize_desmatamento():
    """Carrega e resume os dados de desmatamento."""
    try:
        df = pd.read_parquet(DESMATAMENTO_PATH)
        summarize_df(df, "desmatamento_anual")
        return df
    except FileNotFoundError:
        print(f"Erro: Arquivo {DESMATAMENTO_PATH} não encontrado.")
        return None


def load_and_summarize_ibama_infracoes():
    """
    Carrega, concatena e resume os dados de autos de infração do IBAMA de vários anos.
    """
    # Tomada de decisão: Os dados estão divididos em múltiplos arquivos. Para uma análise
    # completa e temporal, é essencial consolidá-los. Vamos carregar todos os CSVs em um
    # único DataFrame, o que simplifica a exploração e pré-processamento subsequentes.
    csv_files = glob.glob(str(IBAMA_AUTOS_DIR / "auto_infracao_ano_*.csv"))
    if not csv_files:
        print(f"Nenhum arquivo de infração do IBAMA encontrado em {IBAMA_AUTOS_DIR}")
        return None

    dfs = []
    for file in sorted(csv_files):
        try:
            # Tomada de decisão: Assumimos ';' como separador e codificação 'latin1'
            # que é comum para arquivos governamentais mais antigos no Brasil.
            # `low_memory=False` é usado para carregar o arquivo inteiro de uma vez,
            # o que pode prevenir erros de parsing de tipos em arquivos grandes e mistos.
            df = pd.read_csv(file, sep=';', decimal=',', encoding='latin1', low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"Erro ao ler o arquivo {file}: {e}")

    if not dfs:
        print("Nenhum DataFrame do IBAMA foi carregado com sucesso.")
        return None

    ibama_df = pd.concat(dfs, ignore_index=True)
    summarize_df(ibama_df, "ibama_infracoes_consolidadas")
    return ibama_df


def main():
    """
    Função principal para executar a etapa de amostragem (Sampling) do projeto.
    Carrega os dados iniciais, gera e salva resumos para cada fonte de dados.
    """
    print("Iniciando a etapa de Sampling (Amostragem)...")
    setup_reports_dir()
    load_and_summarize_ips()
    load_and_summarize_desmatamento()
    load_and_summarize_ibama_infracoes()
    print("\nEtapa de Sampling concluída.")
    print(f"Verifique os arquivos de resumo no diretório: {REPORTS_DIR.name}")


if __name__ == "__main__":
    main()
