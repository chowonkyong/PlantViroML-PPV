# ==============================================================================
# File: 04_generate_recombination_umap_analysis.py
# Description: Comprehensive Recombination & UMAP Latent Analysis Script (Figure 3)
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
from sklearn.metrics import roc_auc_score, roc_curve
import umap.umap_ as umap

def generate_figure_3():
    print("=== [1/4] 필수 데이터 파일 로드 중 ===")
    res_file = 'integrated_rdp5_full_results.csv'
    emb_csv = 'dnabert2_embeddings.csv'
    
    if not os.path.exists(res_file) or not os.path.exists(emb_csv):
        print(f"[오류] 필수 데이터 파일('{res_file}' 또는 '{emb_csv}')을 찾을 수 없습니다.")
        return

    res_df = pd.read_csv(res_file)
    emb_df = pd.read_csv(emb_csv)
    
    acc_col = next((c for c in emb_df.columns if 'acc' in c.lower() or 'id' in c.lower() or 'seq' in c.lower()), emb_df.columns[0])
    accessions = emb_df[acc_col].values
    embeddings = emb_df.drop(columns=[acc_col]).select_dtypes(include=[np.number]).values
    feature_names = [f"Dim_{i+1}" for i in range(embeddings.shape[1])]

    print("=== [2/4] 재조합주 라벨링, UMAP 및 머신러닝 학습 중 ===")
    recombinant_accs = set()
    for _, row in res_df.iterrows():
        acc_str = str(row['Recombinant_Accessions'])
        for acc in acc_str.split(','):
            cleaned = acc.strip()
            if cleaned:
                recombinant_accs.add(cleaned)

    labels = np.array([1 if any(acc == r or r in acc for r in recombinant_accs) else 0 for acc in accessions])

    # UMAP 차원 축소
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    # Random Forest 학습 및 평가
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.3, random_state=42, stratify=labels if np.sum(labels==1) > 5 else None
    )

    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    y_proba = rf_model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    print("=== [3/4] 6-패널 Figure 3 종합 시각화 생성 중 ===")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 2행 3열 완벽한 대칭형 레이아웃 (총 6개 패널)
    fig = plt.figure(figsize=(19, 11), dpi=600)
    
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    # --- Panel A: Breakpoint Distribution along Genome ---
    breakpoints = []
    for _, row in res_df.iterrows():
        try:
            if pd.notnull(row['Breakpoint_Begin']):
                breakpoints.append(float(row['Breakpoint_Begin']))
            if pd.notnull(row['Breakpoint_End']):
                breakpoints.append(float(row['Breakpoint_End']))
        except:
            continue
            
    sns.histplot(breakpoints, bins=30, ax=ax1, color='#2a9d8f', edgecolor='black', kde=True, alpha=0.7)
    ax1.set_title('Recombination Breakpoint Hotspots', fontsize=11, fontweight='bold', pad=8)
    ax1.set_xlabel('Genome Position (nt)', fontsize=9)
    ax1.set_ylabel('Frequency (Count)', fontsize=9)
    ax1.set_xlim(0, 11000)

    # --- Panel B: Genomic Span of Recombination Events ---
    for idx, row in res_df.iterrows():
        try:
            ev_num = row.get('Event_Num', idx + 1)
            begin = float(row['Breakpoint_Begin'])
            end = float(row['Breakpoint_End'])
            if begin > end: # Circular
                ax2.plot([begin, 10733], [ev_num, ev_num], color='#e76f51', lw=2)
                ax2.plot([0, end], [ev_num, ev_num], color='#e76f51', lw=2)
            else:
                ax2.plot([begin, end], [ev_num, ev_num], color='#264653', lw=2)
        except:
            continue
            
    ax2.set_title('Genomic Span of Events', fontsize=11, fontweight='bold', pad=8)
    ax2.set_xlabel('Genome Position (nt)', fontsize=9)
    ax2.set_ylabel('Recombination Event Number', fontsize=9)
    ax2.set_xlim(0, 11000)

    # --- Panel C: Distribution of Recombinant Fragment Lengths ---
    fragment_lengths = []
    for _, row in res_df.iterrows():
        try:
            begin = float(row['Breakpoint_Begin'])
            end = float(row['Breakpoint_End'])
            length = (end - begin) if end >= begin else (10733 - begin + end)
            if length > 0:
                fragment_lengths.append(length)
        except:
            continue
            
    if fragment_lengths:
        sns.histplot(fragment_lengths, bins=20, ax=ax3, color='#f4a261', edgecolor='black', alpha=0.8)
    ax3.set_title('Recombinant Fragment Lengths', fontsize=11, fontweight='bold', pad=8)
    ax3.set_xlabel('Fragment Length (nt)', fontsize=9)
    ax3.set_ylabel('Event Count', fontsize=9)

    # --- Panel D: DNABERT-2 UMAP Latent Space Separation ---
    ax4.scatter(
        embedding_2d[labels == 0, 0], embedding_2d[labels == 0, 1],
        c='#3b528b', label='Non-recombinant', alpha=0.7, s=20, edgecolor='none'
    )
    ax4.scatter(
        embedding_2d[labels == 1, 0], embedding_2d[labels == 1, 1],
        c='#f1c40f', label='Recombinant', alpha=0.9, s=35, edgecolor='k', linewidth=0.3
    )
    ax4.legend(loc='upper right', frameon=True, fontsize=8, facecolor='white', framealpha=0.9)
    ax4.set_title('DNABERT-2 Latent Space Separation', fontsize=11, fontweight='bold', pad=8)
    ax4.set_xlabel('UMAP Dimension 1', fontsize=9)
    ax4.set_ylabel('UMAP Dimension 2', fontsize=9)

    # --- Panel E: ROC Curve for Recombination ---
    label_text = f'DNABERT-2 (AUC = {auc_score:.3f})'
    ax5.plot(fpr, tpr, color='#2b5c8f', lw=2.5, label=label_text)
    ax5.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    ax5.set_xlim([0.0, 1.0])
    ax5.set_ylim([0.0, 1.05])
    ax5.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=9)
    ax5.set_ylabel('True Positive Rate (Sensitivity)', fontsize=9)
    ax5.set_title('ROC Curve for Recombination', fontsize=11, fontweight='bold', pad=8)
    ax5.legend(loc='lower right', frameon=True, fontsize=9)

    # --- Panel F: Top 10 Feature Importances ---
    importances = rf_model.feature_importances_
    top_indices = np.argsort(importances)[-10:][::-1]
    top_importances = importances[top_indices]
    top_features = [feature_names[i] for i in top_indices]

    sns.barplot(x=top_importances, y=top_features, ax=ax6, palette='crest', edgecolor='black', hue=top_features, legend=False)
    ax6.set_title('Top 10 Latent Dimensions', fontsize=11, fontweight='bold', pad=8)
    ax6.set_xlabel('Feature Importance (Gini)', fontsize=9)
    ax6.set_ylabel('Latent Embedding Dimension', fontsize=9)

    plt.tight_layout()

    # --- [4/4] Figure 3 최종 저장 ---
    png_name = 'Figure3.png'
    pdf_name = 'Figure3.pdf'
    plt.savefig(png_name, dpi=600, bbox_inches='tight')
    plt.savefig(pdf_name, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"\n[완료] 완벽한 6패널 Main Figure 3가 '{png_name}' 및 '{pdf_name}' 파일로 성공적으로 생성되었습니다!")

if __name__ == "__main__":
    generate_figure_3()
