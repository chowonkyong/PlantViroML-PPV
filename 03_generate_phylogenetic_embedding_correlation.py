# ==============================================================================
# File: 03_generate_phylogenetic_embedding_correlation.py
# Description: Phylogenetic vs Embedding Distance Correlation Script (Figure 2)
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, chi2_contingency
from Bio import Phylo

def clean_id(acc_str):
    return str(acc_str).split('|')[0].split('.')[0].strip()

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((k_corr - 1), (r_corr - 1))) if min((k_corr - 1), (r_corr - 1)) > 0 else 0

def generate_figure2():
    print("=== [Figure 2] DNABERT-2 임베딩 거리 및 메타데이터 연관성 패널 생성 중 ===")
    
    # 1. Panel A 데이터 준비 (Mantel / Distance Correlation)
    embeddings_csv = "dnabert2_embeddings.csv"
    if not os.path.exists(embeddings_csv):
        print(f"[오류] '{embeddings_csv}' 파일을 찾을 수 없습니다.")
        return

    df_emb = pd.read_csv(embeddings_csv)
    df_emb['Clean_ID'] = df_emb.iloc[:, 0].apply(clean_id)
    df_emb = df_emb.drop_duplicates(subset=['Clean_ID']).sort_values(by='Clean_ID').reset_index(drop=True)
    emb_accessions = df_emb['Clean_ID'].tolist()
    embeddings = df_emb.iloc[:, 1:-1].values

    emb_dist_vector = pdist(embeddings, metric='cosine')
    emb_dist_matrix = squareform(emb_dist_vector)

    mldist_file = "ppv_full_genomes_trimmed.fasta.mldist"
    tree_taxa = []
    if os.path.exists(mldist_file):
        with open(mldist_file, 'r') as f:
            lines = f.readlines()
        n_taxa = int(lines[0].strip())
        tree_dist_matrix = np.zeros((n_taxa, n_taxa))
        for i in range(n_taxa):
            parts = lines[i+1].strip().split()
            tree_taxa.append(clean_id(parts[0]))
            tree_dist_matrix[i, :] = [float(x) for x in parts[1:]]
    else:
        tree_file = "ppv_full_genomes_trimmed.fasta.treefile"
        tree = Phylo.read(tree_file, "newick")
        tree_taxa = [clean_id(term.name) for term in tree.get_terminals()]
        n = len(tree_taxa)
        tree_dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = tree.distance(tree.get_terminals()[i], tree.get_terminals()[j])
                tree_dist_matrix[i, j] = d
                tree_dist_matrix[j, i] = d

    common_accs = sorted(list(set(emb_accessions).intersection(set(tree_taxa))))
    emb_indices = [emb_accessions.index(acc) for acc in common_accs]
    tree_indices = [tree_taxa.index(acc) for acc in common_accs]

    sub_emb_matrix = emb_dist_matrix[np.ix_(emb_indices, emb_indices)]
    sub_tree_matrix = tree_dist_matrix[np.ix_(tree_indices, tree_indices)]

    i_upper = np.triu_indices(len(common_accs), k=1)
    tree_distances = sub_tree_matrix[i_upper]
    emb_distances = sub_emb_matrix[i_upper]
    corr, p_value = pearsonr(tree_distances, emb_distances)

    # 2. Panel B 데이터 준비 (Metadata Association - Cramér's V)
    meta_file = "ppv_full_genomes_metadata_with_cluster.csv"
    if not os.path.exists(meta_file):
        print(f"[오류] '{meta_file}' 파일을 찾을 수 없습니다.")
        return

    meta_df = pd.read_csv(meta_file)
    target_vars = ['Strain', 'Country', 'Host', 'Year']
    cramers_results = {}
    for var in target_vars:
        if var in meta_df.columns:
            sub_df = meta_df.dropna(subset=[var, 'UMAP_Cluster'])
            sub_df = sub_df[sub_df[var] != 'Unknown']
            ct = pd.crosstab(sub_df['UMAP_Cluster'], sub_df[var])
            cramers_results[var] = cramers_v(ct)

    # 3. 2패널 Figure 생성 (고급스러운 컬러 팔레트 적용)
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=600)

    # Panel A: Scatter Plot & Regression
    ax1 = axes[0]
    sns.regplot(
        x=tree_distances, 
        y=emb_distances, 
        ax=ax1,
        scatter_kws={'alpha': 0.2, 'color': '#2b5c8f', 's': 12},
        line_kws={'color': '#e7298a', 'linewidth': 2.5}
    )
    ax1.set_title("Phylogenetic vs Embedding Distance Correlation", fontsize=13, fontweight="bold")
    ax1.set_xlabel("IQ-TREE Phylogenetic Distance", fontsize=11)
    ax1.set_ylabel("DNABERT-2 Embedding Distance (Cosine)", fontsize=11)
    
    text_str = f"Correlation \(r\) = {corr:.3f}\n\(p\)-value < 0.001"
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax1.text(0.05, 0.95, text_str, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)

    # Panel B: Cramér's V Bar Plot
    ax2 = axes[1]
    vars_list = list(cramers_results.keys())
    vals_list = list(cramers_results.values())
    
    distinct_colors = ['#2b5c8f', '#d95f02', '#7570b3', '#1b9e77']
    bars = ax2.bar(vars_list, vals_list, color=distinct_colors, alpha=0.9, edgecolor='black', width=0.5)

    ax2.set_title("Metadata Association with UMAP Clusters (Cramér's V)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Metadata Category", fontsize=11)
    ax2.set_ylabel("Cramér's V Association Index", fontsize=11)
    ax2.set_ylim(0, 0.7)
    ax2.grid(True, linestyle="--", alpha=0.5, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    output_png = "Figure2.png"
    output_pdf = "Figure2.pdf"
    plt.savefig(output_png, dpi=600, bbox_inches="tight")
    plt.savefig(output_pdf, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"[완료] Figure 2 저장 완료:\n  - PNG: {output_png}\n  - PDF: {output_pdf}")

if __name__ == '__main__':
    generate_figure2()
