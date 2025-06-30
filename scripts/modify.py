import pandas as pd
import numpy as np
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# --- Constantes ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
PROCESSED_DIR = REPORTS_DIR / "processed"

IPS_PATH = DATA_DIR / "ips_2024.csv"
DESMATAMENTO_PATH = DATA_DIR / "desmatamento_anual_municipios.parquet"
IBAMA_AUTOS_DIR = DATA_DIR / "ibama" / "autos_infracao_csv"

# Constantes para análise
ANO_INICIO = 2008  # Ano inicial dos dados de desmatamento
UF_FOCO = 'PA'     # Estado do Pará


def setup_processed_dir():
    """Cria o diretório de saída para os dados processados."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_ips_data():
    """Carrega e processa os dados do IPS."""
    try:
        df = pd.read_csv(IPS_PATH, sep=',', decimal='.')
        # Filtrar apenas municípios do Pará
        df_pa = df[df['UF'] == UF_FOCO].copy()
        
        # Garantir que o código do município seja numérico e consistente
        df_pa['Código IBGE'] = pd.to_numeric(df_pa['Código IBGE'], errors='coerce')
        df_pa.dropna(subset=['Código IBGE'], inplace=True)
        df_pa['Código IBGE'] = df_pa['Código IBGE'].astype('int64')

        print(f"IPS: {len(df_pa)} municípios do Pará carregados")
        return df_pa
    except Exception as e:
        print(f"Erro ao carregar dados do IPS: {e}")
        return None


def load_desmatamento_data():
    """Carrega e processa os dados de desmatamento."""
    try:
        df = pd.read_parquet(DESMATAMENTO_PATH)
        # Filtrar Pará e anos >= 2008
        df_pa = df[(df['ESTADO'] == UF_FOCO) & (df['ano'] >= ANO_INICIO)].copy()
        
        # Converter códigos de município para int64 para facilitar merge
        df_pa['CD_MUN'] = df_pa['CD_MUN'].astype('int64')
        
        print(f"Desmatamento: {len(df_pa)} registros do Pará de {ANO_INICIO} em diante")
        return df_pa
    except Exception as e:
        print(f"Erro ao carregar dados de desmatamento: {e}")
        return None


def load_and_process_ibama_data():
    """Carrega, consolida e processa os dados do IBAMA."""
    try:
        csv_files = glob.glob(str(IBAMA_AUTOS_DIR / "auto_infracao_ano_*.csv"))
        if not csv_files:
            print("Nenhum arquivo do IBAMA encontrado")
            return None
        
        dfs = []
        for file in sorted(csv_files):
            try:
                df = pd.read_csv(file, sep=';', decimal=',', encoding='latin1', low_memory=False)
                dfs.append(df)
            except Exception as e:
                print(f"Erro ao ler {file}: {e}")
        
        if not dfs:
            return None
        
        # Consolidar todos os anos
        ibama_df = pd.concat(dfs, ignore_index=True)
        
        # Filtrar apenas Pará
        ibama_pa = ibama_df[ibama_df['UF'] == UF_FOCO].copy()
        
        # Extrair ano da data de autuação
        ibama_pa['DAT_HORA_AUTO_INFRACAO'] = pd.to_datetime(
            ibama_pa['DAT_HORA_AUTO_INFRACAO'], errors='coerce'
        )
        ibama_pa['ano'] = ibama_pa['DAT_HORA_AUTO_INFRACAO'].dt.year
        
        # Filtrar anos >= 2008
        ibama_pa = ibama_pa[ibama_pa['ano'] >= ANO_INICIO].copy()
        
        # Converter código do município para um tipo numérico consistente
        ibama_pa['COD_MUNICIPIO'] = pd.to_numeric(ibama_pa['COD_MUNICIPIO'], errors='coerce')
        ibama_pa.dropna(subset=['COD_MUNICIPIO'], inplace=True)
        ibama_pa['COD_MUNICIPIO'] = ibama_pa['COD_MUNICIPIO'].astype('int64')
        
        print(f"IBAMA: {len(ibama_pa)} registros do Pará de {ANO_INICIO} em diante")
        return ibama_pa
    except Exception as e:
        print(f"Erro ao carregar dados do IBAMA: {e}")
        return None


def create_ibama_aggregations(ibama_df):
    """
    Cria agregações dos dados do IBAMA por município e ano.
    Tomada de decisão: Separamos infrações de flora das demais para análise específica.
    """
    if ibama_df is None or ibama_df.empty:
        return None
    
    # Identificar infrações de flora
    # Tomada de decisão: Usamos tanto TIPO_INFRACAO quanto DES_INFRACAO para capturar
    # todas as infrações relacionadas à flora/desmatamento
    flora_keywords = ['flora', 'florestal', 'vegetação', 'desmatamento', 'supressão']
    ibama_df['is_flora'] = (
        ibama_df['TIPO_INFRACAO'].str.lower().str.contains('flora', na=False) |
        ibama_df['DES_INFRACAO'].str.lower().str.contains('|'.join(flora_keywords), na=False)
    )
    
    # Agregar por município e ano
    agg_dict = {
        'SEQ_AUTO_INFRACAO': 'count',  # Total de autos
        'VAL_AUTO_INFRACAO': ['sum', 'mean'],  # Valor total e médio das multas
        'is_flora': 'sum',  # Número de autos de flora
    }
    
    ibama_agg = ibama_df.groupby(['COD_MUNICIPIO', 'ano']).agg(agg_dict).reset_index()
    
    # Renomear colunas
    ibama_agg.columns = [
        'COD_MUNICIPIO', 'ano', 
        'ibama_total_autos', 'ibama_valor_total_multas', 'ibama_valor_medio_multas',
        'ibama_autos_flora'
    ]
    
    return ibama_agg


def create_analytical_base_table(df_ips, df_desmatamento, ibama_agg):
    """
    Cria a tabela analítica base combinando todas as fontes de dados.
    A interpolação de dados faltantes do IPS é feita aqui.
    """
    if df_desmatamento is None:
        print("Erro: Dados de desmatamento não disponíveis")
        return None
    
    base_table = df_desmatamento[['CD_MUN', 'NM_MUN', 'ano', 'desmatamento_km2']].copy()
    
    if df_ips is not None:
        # Colunas de IPS para interpolar
        ips_cols_to_interpolate = [
            'Área (km²)', 'População 2022', 'PIB per capita 2021',
            'Índice de Progresso Social', 'Necessidades Humanas Básicas',
            'Fundamentos do Bem-estar', 'Oportunidades'
        ]
        
        # Preparar IPS para merge
        df_ips_merge = df_ips[['Código IBGE'] + ips_cols_to_interpolate].copy()
        df_ips_merge = df_ips_merge.rename(columns={'Código IBGE': 'CD_MUN'})
        
        # Merge com IPS (mantém todos os anos da base_table)
        base_table = base_table.merge(df_ips_merge, on='CD_MUN', how='left')
        
        # Interpolar os dados do IPS para cada município ao longo do tempo
        base_table = base_table.sort_values(by=['CD_MUN', 'ano'])
        
        for col in ips_cols_to_interpolate:
            # Interpolação linear para preencher os 'buracos'
            base_table[col] = base_table.groupby('CD_MUN')[col].transform(
                lambda x: x.interpolate(method='linear')
            )
            # Preenchimento para frente e para trás para os valores nas pontas
            base_table[col] = base_table.groupby('CD_MUN')[col].transform('ffill')
            base_table[col] = base_table.groupby('CD_MUN')[col].transform('bfill')
            
        print(f"Após interpolação do IPS: {len(base_table)} registros")
    
    # Adicionar dados do IBAMA
    if ibama_agg is not None:
        ibama_agg_merge = ibama_agg.rename(columns={'COD_MUNICIPIO': 'CD_MUN'})
        base_table = base_table.merge(ibama_agg_merge, on=['CD_MUN', 'ano'], how='left')
        print(f"Após merge com IBAMA: {len(base_table)} registros")
        
        # Preencher NAs do IBAMA com 0 (significa que não houve autos naquele ano/município)
        ibama_cols = ['ibama_total_autos', 'ibama_valor_total_multas', 
                     'ibama_valor_medio_multas', 'ibama_autos_flora']
        base_table[ibama_cols] = base_table[ibama_cols].fillna(0)
    
    return base_table


def add_temporal_features(df):
    """
    Adiciona features temporais para análise de séries temporais.
    Tomada de decisão: Incluímos features que capturam mudanças políticas e temporais.
    """
    if df is None:
        return None
    
    df = df.copy()
    
    # Features temporais básicas
    df['ano_normalizado'] = df['ano'] - ANO_INICIO  # Anos desde o início (0-based)
    
    # Períodos presidenciais (importante para sua análise de mudanças políticas)
    # Tomada de decisão: Marcamos os períodos presidenciais pois você mencionou
    # que os modelos erram em mudanças de governo
    def get_periodo_presidencial(ano):
        if ano <= 2010:
            return 'Lula'
        elif ano <= 2016:
            return 'Dilma'
        elif ano <= 2018:
            return 'Temer'
        elif ano <= 2022:
            return 'Bolsonaro'
        else:
            return 'Lula'
    
    df['periodo_presidencial'] = df['ano'].apply(get_periodo_presidencial)
    
    # Anos de transição (primeiro ano de cada governo)
    anos_transicao = [2011, 2016, 2019, 2023]  # Dilma, Temer, Bolsonaro, Lula
    df['ano_transicao'] = df['ano'].isin(anos_transicao).astype(int)
    
    # Features de lag para análise temporal
    # Tomada de decisão: Ordenamos por município e ano para calcular lags corretamente
    df = df.sort_values(['CD_MUN', 'ano'])
    
    # Lag de desmatamento (ano anterior)
    df['desmatamento_lag1'] = df.groupby('CD_MUN')['desmatamento_km2'].shift(1)
    
    # Média móvel de 3 anos do desmatamento (usando apenas dados passados para evitar data leakage)
    df['desmatamento_ma3'] = df.groupby('CD_MUN')['desmatamento_km2'].shift(1).rolling(window=3, min_periods=1).mean()
    
    # Taxa de crescimento do desmatamento
    df['desmatamento_crescimento'] = df.groupby('CD_MUN')['desmatamento_km2'].pct_change()
    
    return df


def create_encoding_for_ml(df):
    """
    Prepara encodings para modelos de machine learning.
    """
    if df is None:
        return None
    
    df_encoded = df.copy()
    
    # Encoding para período presidencial
    periodo_mapping = {'Lula': 0, 'Dilma': 1, 'Temer': 2, 'Bolsonaro': 3}
    df_encoded['periodo_presidencial_encoded'] = df_encoded['periodo_presidencial'].map(periodo_mapping)
    
    # Normalizar área do município (importante para comparações)
    if 'Área (km²)' in df_encoded.columns:
        df_encoded['area_normalizada'] = (df_encoded['Área (km²)'] - df_encoded['Área (km²)'].mean()) / df_encoded['Área (km²)'].std()
    
    return df_encoded


def save_analytical_tables(df_base, df_encoded):
    """Salva as tabelas analíticas em diferentes formatos."""
    if df_base is None:
        print("Erro: Tabela analítica base não disponível")
        return
    
    # Tabela base (com todas as colunas originais)
    base_path = PROCESSED_DIR / "analytical_base_table.parquet"
    df_base.to_parquet(base_path, index=False)
    print(f"Tabela analítica base salva: {base_path}")
    
    # Tabela para modelos tabulares (Random Forest, XGBoost)
    if df_encoded is not None:
        # Selecionar apenas colunas numéricas para modelos tabulares
        numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
        df_tabular = df_encoded[numeric_cols].copy()
        
        tabular_path = PROCESSED_DIR / "analytical_table_tabular.parquet"
        df_tabular.to_parquet(tabular_path, index=False)
        print(f"Tabela para modelos tabulares salva: {tabular_path}")
    
    # Tabela para modelos de séries temporais (LSTM, GRU)
    # Tomada de decisão: Ordenamos por município e ano para sequências temporais
    # Usamos df_encoded para incluir features codificadas como periodo_presidencial_encoded
    if df_encoded is not None:
        df_temporal = df_encoded.sort_values(['CD_MUN', 'ano']).copy()
    else:
        df_temporal = df_base.sort_values(['CD_MUN', 'ano']).copy()
    
    temporal_path = PROCESSED_DIR / "analytical_table_temporal.parquet"
    df_temporal.to_parquet(temporal_path, index=False)
    print(f"Tabela para modelos temporais salva: {temporal_path}")
    
    # Salvar resumo das transformações
    summary_path = PROCESSED_DIR / "data_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Resumo da Tabela Analítica Base\n")
        f.write("=" * 50 + "\n\n")
        
        # Usar df_encoded se disponível, senão df_base
        summary_df = df_encoded if df_encoded is not None else df_base
        
        f.write(f"Período: {ANO_INICIO} - {summary_df['ano'].max()}\n")
        f.write(f"Estado: {UF_FOCO}\n")
        f.write(f"Municípios únicos: {summary_df['CD_MUN'].nunique()}\n")
        f.write(f"Total de registros: {len(summary_df)}\n\n")
        
        f.write("Colunas disponíveis:\n")
        for col in summary_df.columns:
            f.write(f"- {col}\n")
        
        f.write(f"\nPeríodos presidenciais incluídos:\n")
        for periodo in summary_df['periodo_presidencial'].unique():
            count = len(summary_df[summary_df['periodo_presidencial'] == periodo])
            f.write(f"- {periodo}: {count} registros\n")
    
    print(f"Resumo salvo: {summary_path}")


def main():
    """
    Função principal para executar a etapa de Modificação (Modify) do SEMMA.
    """
    print("Iniciando a etapa de Modify (Modificação)...")
    print(f"Período de análise: {ANO_INICIO} em diante")
    print(f"Estado: {UF_FOCO}")
    print("-" * 50)
    
    setup_processed_dir()
    
    # Carregar dados
    print("1. Carregando dados...")
    df_ips = load_ips_data()
    df_desmatamento = load_desmatamento_data()
    ibama_df = load_and_process_ibama_data()
    
    # Processar IBAMA
    print("\n2. Processando dados do IBAMA...")
    ibama_agg = create_ibama_aggregations(ibama_df)
    if ibama_agg is not None:
        print(f"IBAMA agregado: {len(ibama_agg)} registros município-ano")
    
    # Criar tabela analítica base
    print("\n3. Criando tabela analítica base...")
    base_table = create_analytical_base_table(df_ips, df_desmatamento, ibama_agg)
    
    if base_table is None:
        print("Erro: Não foi possível criar a tabela analítica base")
        return
    
    # Adicionar features temporais
    print("\n4. Adicionando features temporais...")
    base_table_with_temporal = add_temporal_features(base_table)
    
    # Preparar encoding para ML
    print("\n5. Preparando encodings para ML...")
    encoded_table = create_encoding_for_ml(base_table_with_temporal)
    
    # Salvar tabelas
    print("\n6. Salvando tabelas analíticas...")
    save_analytical_tables(base_table_with_temporal, encoded_table)
    
    print(f"\nEtapa de Modify concluída!")
    print(f"Verifique os arquivos processados em: {PROCESSED_DIR}")
    print("\nArquivos gerados:")
    print("- analytical_base_table.parquet: Tabela completa com todas as features")
    print("- analytical_table_tabular.parquet: Para modelos tabulares (RF, XGBoost)")
    print("- analytical_table_temporal.parquet: Para modelos temporais (LSTM, GRU)")
    print("- data_summary.txt: Resumo dos dados processados")


if __name__ == "__main__":
    main() 