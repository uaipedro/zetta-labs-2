"""
Dashboard Principal - Análise do Desmatamento no Pará
"""
import streamlit as st
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# Adicionar o diretório atual ao path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Importar configurações
from config.settings import PAGE_CONFIG

# Configurar página
st.set_page_config(**PAGE_CONFIG)

# Importar módulos
from data.loader import DataLoader
from components import (
    render_spatial_analysis,
    render_temporal_analysis,
    render_model_analysis
)


def main():
    """Função principal do dashboard"""
    
    # Header do dashboard
    st.markdown("""
    <h1 style='text-align: center; color: #2E8B57;'>
        🌳 Análise do Desmatamento no Pará
    </h1>
    <p style='text-align: center; font-size: 18px; color: #666;'>
        Dashboard interativo com insights acionáveis e previsões baseadas em Machine Learning
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Carregar dados com loading spinner
    with st.spinner("🔄 Carregando dados e modelos..."):
        # Carregar dados base
        df_base, _, predictions = DataLoader.load_base_data()
        
        if df_base is None:
            st.error("❌ Erro ao carregar dados. Verifique se os arquivos estão disponíveis.")
            return
        
        # Carregar dados adicionais
        model_results = DataLoader.load_model_results()
        best_model, best_metadata, best_result = DataLoader.load_best_model()
        feature_importance = DataLoader.load_feature_importance()
        gdf = DataLoader.load_shapefile()
    
    # Sidebar com informações e filtros
    with st.sidebar:
        st.markdown("### 📊 Painel de Controle")
        
        # Informações gerais
        st.markdown("#### 📈 Estatísticas Gerais")
        
        # Período de análise
        min_year = int(df_base['ano'].min())
        max_year = int(df_base['ano'].max())
        st.markdown(f"**Período de Análise:** {min_year}–{max_year}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Municípios", df_base['NM_MUN'].nunique())
        with col2:
            st.metric("Registros", f"{len(df_base):,}")
        
        # Verificar municípios ausentes
        if gdf is not None:
            gdf_mun = set(gdf['NM_MUN'].unique())
            df_mun = set(df_base['NM_MUN'].unique())
            
            missing_mun = gdf_mun - df_mun
            if missing_mun:
                st.warning(f"Município ausente nos dados: **{', '.join(missing_mun)}**")

        st.markdown("---")
        
        
        # Informações adicionais
        with st.expander("ℹ️ Sobre o Dashboard"):
            st.markdown("""
            **Versão:** 2.1.0  
            **Última Atualização:** Dados até 2023
            
            **Fontes de Dados:**
            - PRODES/INPE (Desmatamento)
            - IBAMA (Fiscalização)
            - IPS Amazônia (Indicadores Sociais)
            - IBGE (Dados Socioeconômicos)
            
            **Metodologia:**
            - Modelos: Random Forest e XGBoost
            - Validação: Time Series Split (walk-forward)
            - Features: 
                - Indicadores Socioeconômicos
                - Indicadores de Fiscalização
                - Indicadores de Desmatamento
            """)
    
    # Tabs principais
    tab_names = ["📈 Análise Temporal", "🗺️ Análise Espacial", "🤖 Análise de Modelos"]
    
    # Verificar tab selecionada no session state
    if 'tab' not in st.session_state:
        st.session_state['tab'] = 'temporal'
    
    # Mapear tabs
    tab_map = {
        'temporal': 0,
        'spatial': 1,
        'models': 2
    }
    
    default_index = tab_map.get(st.session_state.get('tab', 'temporal'), 0)
    
    tabs = st.tabs(tab_names)
    
    # Renderizar conteúdo das tabs
    with tabs[0]:
        render_temporal_analysis(
            df_base, 
            model=best_model, 
            metadata=best_metadata, 
            predictions_hist_df=predictions
        )
    
    with tabs[1]:
        render_spatial_analysis(df_base, gdf)
    
    with tabs[2]:
        render_model_analysis(model_results, feature_importance, best_result)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 14px;'>
        <p>Desenvolvido com ❤️ por Pedro Mambelli | 
        Dados atualizados até 2023 | 
        <a href='https://github.com/pedro-mambelli' target='_blank'>GitHub</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 