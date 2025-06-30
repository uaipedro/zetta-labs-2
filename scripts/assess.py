"""
Módulo de avaliação (Assess) - Análise focada e objetiva dos modelos.
Foco em insights acionáveis:
1. Comparação de performance entre modelos ao longo do tempo
2. Importância das variáveis em cada modelo
3. Precisão em anos críticos (2019, 2023)
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pickle
import numpy as np
import json

# Configuração
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CHECKPOINTS_DIR = REPORTS_DIR / "checkpoints"
MODELS_DIR = REPORTS_DIR / "models"
PROCESSED_DIR = REPORTS_DIR / "processed"

# Estilo limpo e profissional
plt.style.use('default')
sns.set_palette(['#2E8B57', '#4169E1'])  # Verde e Azul
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def load_latest_results():
    """Carrega o arquivo de resultados mais recente."""
    checkpoints = list(CHECKPOINTS_DIR.glob("all_results_*.pkl"))
    if not checkpoints:
        print("❌ Nenhum checkpoint encontrado.")
        return None
    
    latest = max(checkpoints, key=os.path.getctime)
    print(f"📊 Carregando: {latest.name}")
    
    with open(latest, 'rb') as f:
        results = pickle.load(f)
    
    # Enriquecer com feature importance dos metadados
    results_df = pd.DataFrame(results)
    results_df = enrich_with_metadata(results_df)
    
    return results_df

def enrich_with_metadata(results_df):
    """Enriquece os dados com informações dos arquivos de metadados."""
    enriched_results = []
    
    for _, row in results_df.iterrows():
        # Carregar metadados do arquivo JSON
        period_dir = MODELS_DIR / f"period_{row['period']:02d}_{row['test_year']}"
        metadata_file = period_dir / f"{row['model_type']}_metadata.json"
        
        row_dict = row.to_dict()
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Adicionar feature importance se disponível
                if 'feature_importance' in metadata:
                    row_dict['feature_importance'] = metadata['feature_importance']
                    
            except Exception as e:
                print(f"⚠️  Erro ao carregar {metadata_file}: {e}")
        
        enriched_results.append(row_dict)
    
    return pd.DataFrame(enriched_results)

def load_predictions(results_df):
    """Carrega predições de todos os períodos."""
    all_preds = []
    
    for _, row in results_df.iterrows():
        period_dir = f"period_{row['period']:02d}_{row['test_year']}"
        pred_file = MODELS_DIR / period_dir / f"{row['model_type']}_predictions.parquet"
        
        if pred_file.exists():
            preds = pd.read_parquet(pred_file)
            preds['model_type'] = row['model_type']
            preds['test_year'] = row['test_year']
            all_preds.append(preds)
    
    return pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

def plot_model_comparison(results_df):
    """Gráfico 1: Comparação direta entre modelos ao longo do tempo."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # R² ao longo do tempo
    for model in results_df['model_type'].unique():
        data = results_df[results_df['model_type'] == model]
        ax1.plot(data['test_year'], data['r2'], 'o-', linewidth=2, markersize=6, 
                label=model.replace('_', ' ').title())
    
    ax1.set_title('R² por Ano de Teste', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Ano')
    ax1.set_ylabel('R²')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # RMSE ao longo do tempo
    for model in results_df['model_type'].unique():
        data = results_df[results_df['model_type'] == model]
        ax2.plot(data['test_year'], data['rmse'], 's-', linewidth=2, markersize=6,
                label=model.replace('_', ' ').title())
    
    ax2.set_title('RMSE por Ano de Teste', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Ano')
    ax2.set_ylabel('RMSE (km²)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = FIGURES_DIR / "model_comparison_timeline.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparação temporal salva: {save_path.name}")
    plt.close()
    
    # Salvar dados para dashboard
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # Remover coluna feature_importance para salvar em Parquet
    results_clean = results_df.drop(columns=['feature_importance'], errors='ignore')
    results_clean.to_parquet(PROCESSED_DIR / "model_comparison_data.parquet")

def plot_feature_importance_comparison(results_df):
    """Gráfico 2: Top 10 features mais importantes por modelo."""
    if 'feature_importance' not in results_df.columns:
        print("⚠️  Feature importance não disponível")
        return
    
    # Verificar se há dados de feature importance
    has_importance = results_df['feature_importance'].apply(lambda x: x is not None and x != {})
    if not has_importance.any():
        print("⚠️  Nenhum dado de feature importance encontrado")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    model_idx = 0
    for model in results_df['model_type'].unique():
        model_data = results_df[results_df['model_type'] == model]
        
        # Agregar importâncias de todos os períodos
        all_importances = {}
        count = 0
        
        for _, row in model_data.iterrows():
            if row['feature_importance'] is not None and row['feature_importance'] != {}:
                feature_imp = row['feature_importance']
                
                # Se for lista de tuplas (nome, valor), converter para dict
                if isinstance(feature_imp, list) and len(feature_imp) > 0 and isinstance(feature_imp[0], list):
                    feature_imp = {item[0]: item[1] for item in feature_imp}
                
                # Agregar importâncias
                for feature, importance in feature_imp.items():
                    if feature not in all_importances:
                        all_importances[feature] = []
                    all_importances[feature].append(importance)
                count += 1
        
        if not all_importances:
            print(f"⚠️  Nenhuma feature importance encontrada para {model}")
            continue
            
        # Calcular média das importâncias
        avg_importances = {feature: np.mean(values) 
                          for feature, values in all_importances.items()}
        
        # Selecionar top 10
        sorted_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)
        top_10 = sorted_features[:10]
        top_10.reverse()  # Para plotar do menor para o maior
        
        # Plot horizontal
        if model_idx < len(axes):
            ax = axes[model_idx]
            features = [item[0] for item in top_10]
            values = [item[1] for item in top_10]
            
            bars = ax.barh(range(len(features)), values, 
                          color=['#2E8B57', '#4169E1'][model_idx], alpha=0.8)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels([name[:25] + '...' if len(name) > 25 else name 
                               for name in features])
            ax.set_title(f'{model.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Importância Média')
            ax.grid(axis='x', alpha=0.3)
            
            # Adicionar valores nas barras
            for j, (bar, value) in enumerate(zip(bars, values)):
                ax.text(value + value*0.01, j, f'{value:.3f}', 
                       va='center', fontsize=9)
            
            print(f"✅ {model}: {count} períodos processados, {len(avg_importances)} features")
        
        model_idx += 1
    
    plt.suptitle('Top 10 Features Mais Importantes por Modelo', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = FIGURES_DIR / "feature_importance_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Importância das features salva: {save_path.name}")
    plt.close()
    
    # Salvar dados das features
    feature_data = []
    for model in results_df['model_type'].unique():
        model_data = results_df[results_df['model_type'] == model]
        
        all_importances = {}
        for _, row in model_data.iterrows():
            if row['feature_importance'] is not None and row['feature_importance'] != {}:
                feature_imp = row['feature_importance']
                
                # Se for lista de tuplas, converter para dict
                if isinstance(feature_imp, list) and len(feature_imp) > 0 and isinstance(feature_imp[0], list):
                    feature_imp = {item[0]: item[1] for item in feature_imp}
                
                for feature, importance in feature_imp.items():
                    if feature not in all_importances:
                        all_importances[feature] = []
                    all_importances[feature].append(importance)
        
        # Calcular médias e salvar
        for feature, values in all_importances.items():
            feature_data.append({
                'model_type': model,
                'feature': feature,
                'importance': np.mean(values)
            })
    
    if feature_data:
        pd.DataFrame(feature_data).to_parquet(PROCESSED_DIR / "feature_importance_data.parquet")

def plot_temporal_evolution(predictions_df):
    """Gráfico 3: Evolução temporal do desmatamento real vs. previsto por modelo."""
    if predictions_df.empty:
        print("⚠️  Dados de predições não disponíveis")
        return
    
    # Agregar dados por ano e modelo
    temporal_data = predictions_df.groupby(['test_year', 'model_type']).agg({
        'y_true': 'mean',
        'y_pred': 'mean'
    }).reset_index()
    
    # Criar dados do real (única linha para ambos os modelos)
    real_data = temporal_data.groupby('test_year')['y_true'].first().reset_index()
    real_data.columns = ['test_year', 'desmatamento_real']
    
    # Criar figura elegante
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Configurar estilo elegante
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    # Plotar linha real (contínua e destacada)
    ax.plot(real_data['test_year'], real_data['desmatamento_real'], 
            color='#2c3e50', linewidth=4, marker='o', markersize=8,
            label='Desmatamento Real', zorder=3, alpha=0.9)
    
    # Plotar predições dos modelos (pontilhadas)
    colors = ['#2E8B57', '#4169E1']  # Verde e Azul
    linestyles = ['--', '-.']
    markers = ['s', '^']
    
    for i, model in enumerate(temporal_data['model_type'].unique()):
        model_data = temporal_data[temporal_data['model_type'] == model]
        model_name = model.replace('_', ' ').title()
        
        ax.plot(model_data['test_year'], model_data['y_pred'],
                color=colors[i], linewidth=3, linestyle=linestyles[i],
                marker=markers[i], markersize=7, alpha=0.8,
                label=f'{model_name} (Previsto)', zorder=2)
    
    # Destacar período de mudança na política ambiental (2019-2022)
    policy_years = [year for year in range(2019, 2023) if year in real_data['test_year'].values]
    
    if policy_years:
        # Área sombreada para todo o período de mudança política
        ax.axvspan(min(policy_years)-0.5, max(policy_years)+0.5, 
                  alpha=0.15, color='gold', zorder=1, 
                  label='Período de Mudança Política')
        
        # Texto explicativo no meio do período
        meio_periodo = (min(policy_years) + max(policy_years)) / 2
        ax.text(meio_periodo, ax.get_ylim()[1]*0.92, 
               'PPCDAm Suspenso\n(2019-2022)', 
               ha='center', va='top', fontsize=10, fontweight='normal',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffacd', 
                        alpha=0.9, edgecolor='#daa520', linewidth=1))
    
    # Destacar anos específicos
    critical_years = [2019, 2023]
    available_critical = [year for year in critical_years if year in real_data['test_year'].values]
    
    for year in available_critical:
        if year == 2019:
            # Linha vertical para marcar mudança política
            ax.axvline(x=year, color='orange', linestyle=':', linewidth=2, alpha=0.7)
            ax.text(year+0.1, ax.get_ylim()[1]*0.82, 'Mudança na\nPolítica Ambiental', 
                   ha='left', va='top', fontsize=9, fontweight='normal',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff8dc', alpha=0.9))
        elif year == 2023:
            # Linha vertical para marcar nova mudança
            ax.axvline(x=year, color='darkgreen', linestyle=':', linewidth=2, alpha=0.7)
            ax.text(year+0.1, ax.get_ylim()[1]*0.82, 'Nova Gestão\nAmbiental', 
                   ha='left', va='top', fontsize=9, fontweight='normal',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0fff0', alpha=0.9))
    
    # Personalizar o gráfico
    ax.set_title('Evolução Temporal do Desmatamento: Observações Durante Mudanças Políticas', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Ano de Teste', fontsize=12, fontweight='bold')
    ax.set_ylabel('Desmatamento Médio (km²)', fontsize=12, fontweight='bold')
    
    # Grade elegante
    ax.grid(True, linestyle='-', alpha=0.2, color='gray')
    ax.set_axisbelow(True)
    
    # Legenda elegante
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, 
                      shadow=True, fontsize=11, title='Séries Temporais')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    legend.get_title().set_fontweight('bold')
    
    # Ajustar limites e ticks
    ax.set_xlim(real_data['test_year'].min() - 0.5, real_data['test_year'].max() + 0.5)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Adicionar estatísticas e contexto no canto
    stats_text = []
    for model in temporal_data['model_type'].unique():
        model_data = temporal_data[temporal_data['model_type'] == model]
        # Calcular correlação entre real e previsto
        correlation = np.corrcoef(model_data['y_true'], model_data['y_pred'])[0,1]
        mae = np.mean(np.abs(model_data['y_true'] - model_data['y_pred']))
        model_name = model.replace('_', ' ').title()
        stats_text.append(f'{model_name}: r={correlation:.3f}, MAE={mae:.1f}')
    
    stats_text.append('')  # Linha em branco
    stats_text.append('Nota: Período destacado coincide com')
    stats_text.append('mudanças na política ambiental brasileira')
    stats_text.append('')
    stats_text.append('PPCDAm: Plano de Prevenção e')
    stats_text.append('Controle do Desmatamento na Amazônia')
    
    stats_box = '\n'.join(stats_text)
    ax.text(0.98, 0.02, stats_box, transform=ax.transAxes, 
           fontsize=9, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    save_path = FIGURES_DIR / "temporal_evolution_real_vs_predicted.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Evolução temporal salva: {save_path.name}")
    plt.close()
    
    # Salvar dados temporais para dashboard
    temporal_data_export = temporal_data.copy()
    temporal_data_export = temporal_data_export.merge(real_data, on='test_year', how='left')
    temporal_data_export.to_parquet(PROCESSED_DIR / "temporal_evolution_data.parquet")

def analyze_critical_years(results_df, predictions_df):
    """Gráfico 4: Performance em anos críticos (2019, 2023)."""
    critical_years = [2019, 2023]
    available_years = [year for year in critical_years if year in results_df['test_year'].values]
    
    if not available_years:
        print("⚠️  Anos críticos (2019, 2023) não encontrados nos dados")
        return
    
    fig, axes = plt.subplots(1, len(available_years), figsize=(6*len(available_years), 5))
    if len(available_years) == 1:
        axes = [axes]
    
    for i, year in enumerate(available_years):
        year_data = results_df[results_df['test_year'] == year]
        
        if year_data.empty:
            continue
            
        # Comparar R² dos modelos neste ano
        models = year_data['model_type'].values
        r2_scores = year_data['r2'].values
        
        bars = axes[i].bar(range(len(models)), r2_scores, 
                          color=['#2E8B57', '#4169E1'][:len(models)], alpha=0.8)
        axes[i].set_xticks(range(len(models)))
        axes[i].set_xticklabels([m.replace('_', ' ').title() for m in models])
        axes[i].set_title(f'Performance em {year}', fontsize=14, fontweight='bold')
        axes[i].set_ylabel('R²')
        axes[i].set_ylim(0, 1)
        axes[i].grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, score in zip(bars, r2_scores):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Adicionar contexto do ano
        if year == 2019:
            axes[i].text(0.5, 0.95, 'Início de mudanças na política ambiental', 
                        transform=axes[i].transAxes, ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='#fffacd', alpha=0.8))
        elif year == 2023:
            axes[i].text(0.5, 0.95, 'Nova gestão ambiental', 
                        transform=axes[i].transAxes, ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='#f0fff0', alpha=0.8))
    
    plt.suptitle('Performance em Anos Críticos', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = FIGURES_DIR / "critical_years_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Análise de anos críticos salva: {save_path.name}")
    plt.close()
    
    # Salvar dados dos anos críticos
    critical_data = results_df[results_df['test_year'].isin(available_years)].copy()
    critical_data_clean = critical_data.drop(columns=['feature_importance'], errors='ignore')
    critical_data_clean.to_parquet(PROCESSED_DIR / "critical_years_data.parquet")

def create_summary_table(results_df):
    """Tabela resumo com métricas principais."""
    summary = results_df.groupby('model_type').agg({
        'r2': ['mean', 'std', 'min', 'max'],
        'rmse': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    # Achatar colunas
    summary.columns = [f'{metric}_{stat}' for metric, stat in summary.columns]
    summary = summary.reset_index()
    
    # Adicionar ranking
    summary['r2_rank'] = summary['r2_mean'].rank(ascending=False).astype(int)
    summary['rmse_rank'] = summary['rmse_mean'].rank(ascending=True).astype(int)
    
    # Salvar
    summary.to_parquet(PROCESSED_DIR / "model_summary.parquet")
    
    print("\n📊 RESUMO DOS MODELOS:")
    print("=" * 50)
    for _, row in summary.iterrows():
        model = row['model_type'].replace('_', ' ').title()
        print(f"{model}:")
        print(f"  R² médio: {row['r2_mean']:.3f} (±{row['r2_std']:.3f}) - Ranking: #{row['r2_rank']}")
        print(f"  RMSE médio: {row['rmse_mean']:.1f} (±{row['rmse_std']:.1f}) - Ranking: #{row['rmse_rank']}")
        print()

def main():
    """Análise focada e objetiva."""
    print("\n🎯 ANÁLISE FOCADA DE MODELOS")
    print("=" * 40)
    
    # Carregar dados
    results_df = load_latest_results()
    if results_df is None:
        return
    
    predictions_df = load_predictions(results_df)
    print(f"📈 {len(results_df)} resultados | {len(predictions_df):,} predições")
    
    # Criar diretórios
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n🔍 Gerando análises...")
    
    # 1. Comparação temporal entre modelos
    plot_model_comparison(results_df)
    
    # 2. Importância das features
    plot_feature_importance_comparison(results_df)
    
    # 3. Evolução temporal: Real vs Previsto
    plot_temporal_evolution(predictions_df)
    
    # 4. Anos críticos
    analyze_critical_years(results_df, predictions_df)
    
    # 5. Resumo executivo
    create_summary_table(results_df)
    
    print(f"\n✅ Análise concluída!")
    print(f"📁 Gráficos: {FIGURES_DIR}")
    print(f"💾 Dados: {PROCESSED_DIR}")

if __name__ == "__main__":
    main() 