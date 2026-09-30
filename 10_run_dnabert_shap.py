# ==============================================================================
# File: 10_run_dnabert_shap.py
# Description: Detailed DNABERT-2 Latent Space SHAP Analysis Script
# ==============================================================================

import matplotlib
matplotlib.use('Agg') # GUI 창 없이 파일로 저장

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

def run_shap_analysis():
    print("=== [1/4] 데이터 및 메타데이터 로드 중 ===")
    emb_csv = 'dnabert2_embeddings.csv'
    meta_csv = 'ppv_full_genomes_metadata_with_cluster.csv'
    
    if not os.path.exists(emb_csv) or not os.path.exists(meta_csv):
        print("[오류] 임베딩 파일 또는 메타데이터 파일을 찾을 수 없습니다.")
        return

    emb_df = pd.read_csv(emb_csv)
    meta_df = pd.read_csv(meta_csv)

    acc_col = next((c for c in emb_df.columns if 'acc' in c.lower() or 'id' in c.lower() or 'seq' in c.lower()), emb_df.columns[0])
    
    # 수치형 임베딩(768차원)만 정밀 추출
    embeddings = emb_df.select_dtypes(include=[np.number]).values
    if embeddings.shape[1] != 768:
        embedding_cols = [c for c in emb_df.columns if c != acc_col and emb_df[c].dtype in [np.float64, np.float32, np.int64, float, int]]
        embeddings = emb_df[embedding_cols].values
        
    feature_names = [f"Dim_{i+1}" for i in range(embeddings.shape[1])]
    print(f"추출된 임베딩 형태: {embeddings.shape}")

    meta_acc_col = next((c for c in meta_df.columns if 'acc' in c.lower() or 'id' in c.lower() or 'seq' in c.lower()), meta_df.columns[0])
    strain_col = next((c for c in meta_df.columns if 'strain' in c.lower()), None)
    host_col = next((c for c in meta_df.columns if 'host' in c.lower()), None)

    if not strain_col or not host_col:
        print(f"[오류] 메타데이터 내 Strain/Host 컬럼을 찾을 수 없습니다. (컬럼: {list(meta_df.columns)})")
        return

    # 병합
    merged_df = pd.merge(emb_df, meta_df, left_on=acc_col, right_on=meta_acc_col, how='inner')
    merged_indices = emb_df[emb_df[acc_col].isin(merged_df[acc_col])].index.values
    
    X_all = embeddings[merged_indices]
    strains = merged_df[strain_col].values
    hosts = merged_df[host_col].values

    print(f"병합 완료된 유효 샘플 수: {len(X_all)}개")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # --- Task 1: Strain 분류 (M vs D) ---
    print("=== [2/4] Task 1: Strain 분류 (M vs D) SHAP 분석 ===")
    mask_strain = np.isin(strains, ['M', 'D'])
    if mask_strain.sum() > 10:
        X_strain = X_all[mask_strain]
        y_strain = (strains[mask_strain] == 'M').astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X_strain, y_strain, test_size=0.3, random_state=42, stratify=y_strain)
        rf_strain = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        rf_strain.fit(X_train, y_train)
        
        print(f"Strain (M vs D) 모델 Test Accuracy: {rf_strain.score(X_test, y_test):.3f}")
        
        explainer = shap.TreeExplainer(rf_strain)
        shap_values = explainer.shap_values(X_test)
        s_vals = shap_values[1] if isinstance(shap_values, list) else shap_values

        mean_shap = np.mean(np.abs(s_vals), axis=0)
        if mean_shap.ndim > 1:
            mean_shap = mean_shap.flatten()
            
        top_indices = np.argsort(mean_shap)[-10:][::-1]
        top_indices = [int(i) for i in top_indices if 0 <= i < len(feature_names)]
        top_importance = mean_shap[top_indices]
        top_features = [feature_names[i] for i in top_indices]

        plt.figure(figsize=(10, 6), dpi=600)
        sns.barplot(x=top_importance, y=top_features, palette='crest', edgecolor='black', hue=top_features, legend=False)
        plt.title('Top 10 Latent Dimensions for Strain (M vs D) - SHAP', fontsize=13, fontweight='bold', pad=12)
        plt.xlabel('Mean Absolute SHAP Value (Impact on Model Output)', fontsize=11)
        plt.ylabel('DNABERT-2 Latent Dimension', fontsize=11)
        plt.tight_layout()
        plt.savefig('shap_bar_strain_M_vs_D.png', dpi=600, bbox_inches='tight')
        plt.savefig('shap_bar_strain_M_vs_D.pdf', dpi=600, bbox_inches='tight')
        plt.close()
        print("Strain SHAP 바 플롯 저장 완료: 'shap_bar_strain_M_vs_D.png/.pdf'")

    # --- Task 2: Host 적응성 분류 (Prunus mume vs Prunus domestica) ---
    print("=== [3/4] Task 2: Host 적응성 분류 (Prunus mume vs Prunus domestica) SHAP 분석 ===")
    mask_host = np.isin(hosts, ['Prunus mume', 'Prunus domestica'])
    if mask_host.sum() > 10:
        X_host = X_all[mask_host]
        y_host = (hosts[mask_host] == 'Prunus mume').astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X_host, y_host, test_size=0.3, random_state=42, stratify=y_host)
        rf_host = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        rf_host.fit(X_train, y_train)
        
        print(f"Host 적응성 모델 Test Accuracy: {rf_host.score(X_test, y_test):.3f}")
        
        explainer = shap.TreeExplainer(rf_host)
        shap_values = explainer.shap_values(X_test)
        s_vals = shap_values[1] if isinstance(shap_values, list) else shap_values

        mean_shap = np.mean(np.abs(s_vals), axis=0)
        if mean_shap.ndim > 1:
            mean_shap = mean_shap.flatten()
            
        top_indices = np.argsort(mean_shap)[-10:][::-1]
        top_indices = [int(i) for i in top_indices if 0 <= i < len(feature_names)]
        top_importance = mean_shap[top_indices]
        top_features = [feature_names[i] for i in top_indices]

        plt.figure(figsize=(10, 6), dpi=600)
        sns.barplot(x=top_importance, y=top_features, palette='crest', edgecolor='black', hue=top_features, legend=False)
        plt.title('Top 10 Latent Dimensions for Host (Mume vs Domestica) - SHAP', fontsize=13, fontweight='bold', pad=12)
        plt.xlabel('Mean Absolute SHAP Value (Impact on Model Output)', fontsize=11)
        plt.ylabel('DNABERT-2 Latent Dimension', fontsize=11)
        plt.tight_layout()
        plt.savefig('shap_bar_host_mume_vs_domestica.png', dpi=600, bbox_inches='tight')
        plt.savefig('shap_bar_host_mume_vs_domestica.pdf', dpi=600, bbox_inches='tight')
        plt.close()
        print("Host SHAP 바 플롯 저장 완료: 'shap_bar_host_mume_vs_domestica.png/.pdf'")

    print("=== [4/4] 완료 ===")

if __name__ == "__main__":
    run_shap_analysis()
