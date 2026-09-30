# ==============================================================================
# File: 06_generate_phylodynamic_molecular_clock.py
# Description: Phylodynamic Molecular Clock & AI Latent Divergence Analysis Script (Figure 5)
# ==============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy import stats
from Bio import Phylo

print("=== [1/4] 데이터 및 divergence_tree 로딩 중 ===")
treetime_dir = "treetime_output_strict" if os.path.exists("treetime_output_strict") else "."
nexus_path = os.path.join(treetime_dir, "divergence_tree.nexus")
if not os.path.exists(nexus_path):
    nexus_path = os.path.join(treetime_dir, "timetree.nexus")

dates_path = os.path.join(treetime_dir, "dates.tsv")
meta_file = "ppv_full_genomes_metadata_cleaned.csv"
ai_summary_file = "dnabert2_embeddings_summary.csv"

dates_df = pd.read_csv(dates_path, sep='\t')
meta_df = pd.read_csv(meta_file)
ai_df = pd.read_csv(ai_summary_file)

dates_df.columns = [c.strip() for c in dates_df.columns]
meta_df.columns = [c.strip() for c in meta_df.columns]
ai_df.columns = [c.strip() for c in ai_df.columns]

# ID 정제
id_col_dates = '#node' if '#node' in dates_df.columns else dates_df.columns[0]
dates_df['clean_id'] = dates_df[id_col_dates].astype(str).str.strip()

meta_id_col = 'Accession' if 'Accession' in meta_df.columns else meta_df.columns[0]
meta_df['clean_id'] = meta_df[meta_id_col].astype(str).str.strip()

ai_id_col = 'Accession' if 'Accession' in ai_df.columns else ai_df.columns[0]
ai_df['clean_id'] = ai_df[ai_id_col].astype(str).str.strip()

# Strain 컬럼 매칭
strain_col = None
for c in meta_df.columns:
    if c.lower() == 'strain':
        strain_col = c
        break
if not strain_col:
    strain_col = [c for c in meta_df.columns if 'strain' in c.lower()][0]
meta_df['Strain'] = meta_df[strain_col]

# 루트에서 팁까지의 정확한 돌연변이 거리(Root-to-tip distance) 계산
tree = Phylo.read(nexus_path, "nexus")
root = tree.root
div_dict = {}
for term in tree.get_terminals():
    dist = tree.distance(root, term)
    div_dict[term.name.strip()] = dist

dates_df['rtt_distance'] = dates_df['clean_id'].map(div_dict)
dates_df['numeric date'] = pd.to_numeric(dates_df['numeric date'], errors='coerce')

# 데이터 병합
ai_subset = ai_df[['clean_id', 'AI_Embedding_Divergence']].copy()
meta_subset = meta_df[['clean_id', 'Strain']].copy()
dates_subset = dates_df[['clean_id', 'numeric date', 'rtt_distance']].copy()

merged = pd.merge(ai_subset, meta_subset, on='clean_id', how='inner')
merged = pd.merge(merged, dates_subset, on='clean_id', how='inner')

clean_df = merged.dropna(subset=['AI_Embedding_Divergence', 'Strain', 'rtt_distance', 'numeric date']).copy()
print(f"-> 최종 유효 분석 샘플 수: {len(clean_df)}개")

print("=== [2/4] 분자시계 회귀 통계 계산 중 ===")
x = clean_df['numeric date']
y = clean_df['rtt_distance']
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
root_date = -intercept / slope if slope != 0 else np.nan
print(f"-> 회귀 분석 결과: 기울기(β) = {slope:.2e}, Root date = {root_date:.1f}")

print("=== [3/4] Figure 5 (4-Panel) 600 DPI 고해상도 피겨 생성 중 ===")
matplotlib.use('Agg')
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=600)

ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

unique_strains = clean_df['Strain'].unique()
palette = sns.color_palette("Set2", len(unique_strains))
strain_color_map = dict(zip(unique_strains, palette))

spot_size = 75

# [Panel A] Molecular Clock Root-to-Tip Regression
for strain, color in strain_color_map.items():
    sub = clean_df[clean_df['Strain'] == strain]
    ax1.scatter(sub['numeric date'], sub['rtt_distance'], label=strain, color=color, alpha=0.85, s=spot_size, edgecolors='w', linewidths=0.6)

x_vals = np.linspace(x.min(), x.max(), 100)
y_vals = intercept + slope * x_vals
ax1.plot(x_vals, y_vals, color='tab:blue', linewidth=2.5, label='Regression')

stats_text = f"y = α + βt\nβ = {slope:.2e}\nRoot date: {root_date:.1f}"
# 통계 텍스트를 좌측 상단에 배치
ax1.text(0.03, 0.95, stats_text, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))

ax1.set_title("Molecular Clock (Root-to-Tip Regression)", fontsize=13, fontweight='bold')
ax1.set_xlabel("Date", fontsize=11)
ax1.set_ylabel("Root-to-Tip Distance", fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.5)

# 범례를 플롯 내부가 아닌 바깥쪽 우측 상단으로 이동하여 겹침 방지
ax1.legend(title="Strain", fontsize=8, title_fontsize=9, loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=True)

# [Panel B] AI Latent Space Divergence (Boxplot)
sns.boxplot(x='Strain', y='AI_Embedding_Divergence', data=clean_df, ax=ax2, palette=strain_color_map, fliersize=4)
ax2.set_title("AI Latent Space Divergence (DNABERT-2)", fontsize=13, fontweight='bold')
ax2.set_xlabel("PPV Strain", fontsize=11)
ax2.set_ylabel("AI Embedding Distance", fontsize=11)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

# [Panel C] Evolutionary vs. AI Latent Divergence
for strain, color in strain_color_map.items():
    sub = clean_df[clean_df['Strain'] == strain]
    ax3.scatter(sub['rtt_distance'], sub['AI_Embedding_Divergence'], label=strain, color=color, alpha=0.85, s=spot_size, edgecolors='w', linewidths=0.4)

ax3.set_title("Evolutionary vs. AI Latent Divergence", fontsize=13, fontweight='bold')
ax3.set_xlabel("TreeTime Evolutionary Divergence", fontsize=11)
ax3.set_ylabel("DNABERT-2 Embedding Divergence", fontsize=11)
ax3.grid(True, linestyle='--', alpha=0.5)

# [Panel D] Temporal Trend of AI Divergence
for strain, color in strain_color_map.items():
    sub = clean_df[clean_df['Strain'] == strain]
    ax4.scatter(sub['numeric date'], sub['AI_Embedding_Divergence'], label=strain, color=color, alpha=0.85, s=spot_size, edgecolors='w', linewidths=0.4)

ax4.set_title("Temporal Trend of AI Latent Divergence", fontsize=13, fontweight='bold')
ax4.set_xlabel("Estimated Isolation Year", fontsize=11)
ax4.set_ylabel("AI Embedding Divergence", fontsize=11)
ax4.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

# 600 DPI 고해상도 PNG 및 PDF 동시 출력
output_pdf = "Figure5.pdf"
output_png = "Figure5.png"
plt.savefig(output_pdf, format='pdf', dpi=600, bbox_inches='tight')
plt.savefig(output_png, format='png', dpi=600, bbox_inches='tight')
plt.close()

print(f"[성공] Figure 5 범례 겹침 개선 및 타이틀 번호 제거 완료:\n -> {output_pdf}\n -> {output_png}")
