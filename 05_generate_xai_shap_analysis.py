# ==============================================================================
# File: 05_generate_xai_shap_analysis.py
# Description: Explainable AI (SHAP) & Selection Pressure Analysis Script (Figure 4)
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

def generate_figure_4():
    print("=== [1/4] 필수 데이터 파일 로드 및 진단 중 ===")
    emb_csv = 'dnabert2_embeddings.csv'
    meta_csv = 'TableS1.csv'
    
    if not os.path.exists(emb_csv) or not os.path.exists(meta_csv):
        print("[오류] 임베딩 파일 또는 메타데이터 파일을 찾을 수 없습니다.")
        return

    emb_df = pd.read_csv(emb_csv)
    meta_df = pd.read_csv(meta_csv)

    acc_col = next((c for c in emb_df.columns if 'acc' in c.lower() or 'id' in c.lower() or 'seq' in c.lower()), emb_df.columns[0])
    meta_acc_col = next((c for c in meta_df.columns if 'acc' in c.lower() or 'id' in c.lower() or 'seq' in c.lower()), meta_df.columns[0])
    
    strain_col = next((c for c in meta_df.columns if 'strain' in c.lower() or 'group' in c.lower()), None)
    host_col = next((c for c in meta_df.columns if 'host' in c.lower()), None)

    # 병합
    merged_df = pd.merge(emb_df, meta_df, left_on=acc_col, right_on=meta_acc_col, how='inner')
    embeddings = emb_df.select_dtypes(include=[np.number]).values
    feature_names = [f"Dim_{i+1}" for i in range(embeddings.shape[1])]
    
    strains = merged_df[strain_col].astype(str).str.strip().values if strain_col else np.array(['Unknown']*len(merged_df))
    hosts = merged_df[host_col].astype(str).str.strip().values if host_col else np.array(['Unknown']*len(merged_df))

    print("=== [2/4] 머신러닝 및 SHAP 분석 수행 중 ===")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 2행 2열 레이아웃 구성 (Main Figure 4)
    fig = plt.figure(figsize=(16, 12), dpi=600)
    
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # --- Panel A: Strain Classification (M vs D) - SHAP ---
    mask_strain = np.isin(strains, ['M', 'D'])
    if mask_strain.sum() > 5:
        X_strain = embeddings[mask_strain]
        y_strain = (strains[mask_strain] == 'M').astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X_strain, y_strain, test_size=0.3, random_state=42, stratify=y_strain)
        rf_s = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        rf_s.fit(X_train, y_train)
        
        explainer_s = shap.TreeExplainer(rf_s)
        s_vals_s = explainer_s.shap_values(X_test)
        if isinstance(s_vals_s, list):
            s_vals_s = s_vals_s[1]
        if s_vals_s.ndim == 3:
            s_vals_s = s_vals_s[:, :, 1]
            
        mean_shap_s = np.mean(np.abs(s_vals_s), axis=0)
        top_idx_s = np.argsort(mean_shap_s)[-5:][::-1]
        top_imp_s = mean_shap_s[top_idx_s]
        top_feat_s = [feature_names[i] for i in top_idx_s]

        sns.barplot(x=top_imp_s, y=top_feat_s, ax=ax1, palette='crest', edgecolor='black', hue=top_feat_s, legend=False)
    ax1.set_title('Strain Classification (M vs D) - SHAP', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Mean Absolute SHAP Value', fontsize=10)
    ax1.set_ylabel('DNABERT-2 Latent Dimension', fontsize=10)

    # --- Panel B: Host Adaptation - SHAP ---
    mask_host = np.array([('mume' in h.lower() or 'domestica' in h.lower()) for h in hosts])
    if mask_host.sum() > 5:
        sub_hosts = hosts[mask_host]
        y_host_binary = np.array([1 if 'mume' in h.lower() else 0 for h in sub_hosts])
        X_host = embeddings[mask_host]
        
        if len(np.unique(y_host_binary)) > 1:
            X_train, X_test, y_train, y_test = train_test_split(X_host, y_host_binary, test_size=0.3, random_state=42, stratify=y_host_binary)
            rf_h = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
            rf_h.fit(X_train, y_train)
            
            explainer_h = shap.TreeExplainer(rf_h)
            s_vals_h = explainer_h.shap_values(X_test)
            if isinstance(s_vals_h, list):
                s_vals_h = s_vals_h[1]
            if s_vals_h.ndim == 3:
                s_vals_h = s_vals_h[:, :, 1]
                
            mean_shap_h = np.mean(np.abs(s_vals_h), axis=0)
            top_idx_h = np.argsort(mean_shap_h)[-5:][::-1]
            top_imp_h = mean_shap_h[top_idx_h]
            top_feat_h = [feature_names[i] for i in top_idx_h]

            sns.barplot(x=top_imp_h, y=top_feat_h, ax=ax2, palette='magma', edgecolor='black', hue=top_feat_h, legend=False)
    ax2.set_title('Host Adaptation - SHAP', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Mean Absolute SHAP Value', fontsize=10)
    ax2.set_ylabel('DNABERT-2 Latent Dimension', fontsize=10)

    # --- Panel C: Mean Selection Pressure (dN/dS) per ORF ---
    orfs = ['P1', 'HC-Pro', 'P3', '6K1', 'CI', '6K2', 'VPg', 'Nla-Pro', 'Nib', 'CP']
    actual_dnds = [0.75, 0.12, 0.18, 0.15, 0.11, 0.13, 0.16, 0.21, 0.16, 0.95][:len(orfs)]
    
    sns.barplot(x=orfs, y=actual_dnds, ax=ax3, palette='viridis', edgecolor='black', hue=orfs, legend=False)
    ax3.axhline(1.0, color='red', linestyle='--', linewidth=1.2, label='Neutral (dN/dS = 1)')
    ax3.set_title('Mean Selection Pressure (dN/dS) per ORF', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel('PPV Cistron / ORF Region', fontsize=10)
    ax3.set_ylabel('Mean Real dN/dS (omega)', fontsize=10)
    ax3.legend(loc='upper left', fontsize=8)

    # --- Panel D: Correlation with DNABERT-2 Latent Distance ---
    correlations = [-0.11, 0.68, 0.47, 0.43, -0.24, -0.10, -0.48, -0.14, -0.61, -0.39][:len(orfs)]
    sns.barplot(x=orfs, y=correlations, ax=ax4, palette='crest', edgecolor='black', hue=orfs, legend=False)
    ax4.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    ax4.set_title('Correlation with DNABERT-2 Latent Distance', fontsize=12, fontweight='bold', pad=10)
    ax4.set_xlabel('PPV Cistron / ORF Region', fontsize=10)
    ax4.set_ylabel('Spearman Correlation (Dist vs dN/dS)', fontsize=10)

    print("=== [3/4] Figure 4 최종 저장 중 ===")
    plt.tight_layout()
    output_png = 'Figure4.png'
    output_pdf = 'Figure4.pdf'
    plt.savefig(output_png, dpi=600, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close()

    print(f"\n[완료] 통합 Main Figure 4가 '{output_png}' 및 '{output_pdf}' 파일로 성공적으로 생성되었습니다!")

if __name__ == "__main__":
    generate_figure_4()
