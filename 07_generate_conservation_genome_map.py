# ==============================================================================
# File: 07_generate_conservation_genome_map.py
# Description: Comprehensive Conservation & Genome Organization Script (Figure 7)
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Bio import AlignIO

def run_comprehensive_analysis():
    print("=== Step 1: Loading Alignment and Metadata ===")
    alignment_file = 'ppv_full_genomes_aligned.fasta'
    if not os.path.exists(alignment_file):
        alignment_file = 'ppv_full_genomes_trimmed.fasta'
        
    metadata_path = 'ppv_full_genomes_metadata_cleaned.csv'
    if not os.path.exists(metadata_path):
        metadata_path = 'ppv_full_genomes_with_cluster.csv'
        
    alignment = AlignIO.read(alignment_file, "fasta")
    metadata = pd.read_csv(metadata_path)
    
    align_dict = {}
    for rec in alignment:
        clean_id = rec.id.strip()
        align_dict[clean_id] = str(rec.seq).upper()
        short_id = clean_id.split()[0].split('|')[-1]
        align_dict[short_id] = str(rec.seq).upper()

    id_col_meta = None
    for col in ['strain', 'strain_name', 'sequence_name', 'accession', 'id', 'Unnamed: 0', 'name']:
        if col in metadata.columns:
            id_col_meta = col
            break

    print("=== Step 2: Investigating Major Groups (Hosts, Countries, Strains) ===")
    host_col = next((c for c in ['host', 'Host', 'host_species', 'Species'] if c in metadata.columns), None)
    country_col = next((c for c in ['country', 'Country', 'geographic_region', 'Location'] if c in metadata.columns), None)
    strain_col = next((c for c in ['strain_type', 'Strain', 'group', 'cluster', 'PPV_strain'] if c in metadata.columns), None)

    top_hosts = metadata[host_col].value_counts().head(3).index.tolist() if host_col else []
    top_countries = metadata[country_col].value_counts().head(3).index.tolist() if country_col else []
    top_strains = metadata[strain_col].value_counts().head(4).index.tolist() if strain_col else []

    window_size = 100
    step_size = 20
    alignment_length = alignment.get_alignment_length()

    def compute_entropy_profile(group_seqs):
        group_array = np.array([list(seq) for seq in group_seqs])
        align_len = group_array.shape[1]
        windows, scores = [], []
        for start in range(0, align_len - window_size + 1, step_size):
            end = start + window_size
            window_cols = group_array[:, start:end]
            col_entropies = []
            for col_idx in range(window_cols.shape[1]):
                col = window_cols[:, col_idx]
                valid_col = col[(col == 'A') | (col == 'C') | (col == 'G') | (col == 'T')]
                if len(valid_col) == 0:
                    col_entropies.append(0)
                    continue
                _, counts = np.unique(valid_col, return_counts=True)
                freqs = counts / len(valid_col)
                entropy = -np.sum(freqs * np.log2(freqs + 1e-9))
                col_entropies.append(entropy)
            windows.append(start + (window_size // 2))
            scores.append(np.mean(col_entropies))
        return np.array(windows), np.array(scores)

    def get_seqs_for_group(col_name, val):
        sub_meta = metadata[metadata[col_name] == val]
        seqs = []
        for _, r in sub_meta.iterrows():
            if id_col_meta:
                val_id = str(r[id_col_meta]).strip()
                if val_id in align_dict:
                    seqs.append(align_dict[val_id])
                    continue
                elif val_id.split()[0] in align_dict:
                    seqs.append(align_dict[val_id.split()[0]])
                    continue
            for v in r.values:
                str_v = str(v).strip()
                if str_v in align_dict:
                    seqs.append(align_dict[str_v])
                    break
        return seqs

    print("=== Step 3: Generating Figure 7 (Multi-panel Conservation & Genome Map) ===")
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1, 0.35]})
    
    # Panel A
    all_seqs = list(align_dict.values())
    all_wins, all_scores = compute_entropy_profile(all_seqs)
    threshold = np.percentile(all_scores, 5)

    axes[0].plot(all_wins, all_scores, color='#2b5c8f', linewidth=1.5, label='Sequence Variability')
    axes[0].axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label='Top 5% Threshold')
    axes[0].set_title('Genome-wide Sliding Window Scan & Ultra-conserved Regions', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Entropy', fontsize=10)
    axes[0].legend(loc='upper right', fontsize=9)

    # Panel B
    if host_col:
        colors = ['#2b5c8f', '#d95f02', '#7570b3']
        for idx, host in enumerate(top_hosts):
            seqs = get_seqs_for_group(host_col, host)
            if len(seqs) >= 2:
                w, s = compute_entropy_profile(seqs)
                axes[1].plot(w, s, label=f'Host: {host} (n={len(seqs)})', color=colors[idx % len(colors)], linewidth=1.4)
    axes[1].set_title('Comparative Variability Across Major Prunus Hosts', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Entropy', fontsize=10)
    axes[1].legend(loc='upper right', fontsize=9)

    # Panel C
    if country_col:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for idx, country in enumerate(top_countries):
            seqs = get_seqs_for_group(country_col, country)
            if len(seqs) >= 2:
                w, s = compute_entropy_profile(seqs)
                axes[2].plot(w, s, label=f'Country: {country} (n={len(seqs)})', color=colors[idx % len(colors)], linewidth=1.4)
    axes[2].set_title('Comparative Variability Across Major Countries', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Entropy', fontsize=10)
    axes[2].legend(loc='upper right', fontsize=9)

    # Panel D
    if strain_col:
        colors = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        for idx, strain in enumerate(top_strains):
            seqs = get_seqs_for_group(strain_col, strain)
            if len(seqs) >= 2:
                w, s = compute_entropy_profile(seqs)
                axes[3].plot(w, s, label=f'Strain: {strain} (n={len(seqs)})', color=colors[idx % len(colors)], linewidth=1.4)
    axes[3].set_title('Comparative Variability Across Major PPV Strains', fontsize=11, fontweight='bold')
    axes[3].set_ylabel('Entropy', fontsize=10)
    axes[3].legend(loc='upper right', fontsize=9)

    # Panel E: PPV Genome Organization
    genes = [
        ("P1", 150, 950, "#a6cee3"),
        ("HC-Pro", 951, 2400, "#1f78b4"),
        ("P3", 2401, 3420, "#b2df8a"),
        ("6K1", 3421, 3580, "#33a02c"),
        ("CI", 3581, 5470, "#fb9a99"),
        ("6K2", 5471, 5630, "#e31a1c"),
        ("VPg", 5631, 6300, "#fdbf6f"),
        ("Pro", 6301, 7000, "#ff7f00"),
        ("NIb", 7001, 8560, "#cab2d6"),
        ("CP", 8561, 9480, "#6a3d9a")
    ]
    
    axes[4].set_ylim(-0.2, 1.2)
    axes[4].set_xlim(0, alignment_length)
    axes[4].axis('off')
    axes[4].plot([0, alignment_length], [0.4, 0.4], color='black', linewidth=1.5, zorder=1)
    
    for name, start, end, col in genes:
        rect = patches.Rectangle((start, 0.0), end - start, 0.8, facecolor=col, edgecolor='black', linewidth=0.8, zorder=2)
        axes[4].add_patch(rect)
        axes[4].text((start + end) / 2, 0.4, name, ha='center', va='center', fontsize=9.5, fontweight='bold', color='black')

    axes[4].set_title('PPV Genome Organization', fontsize=11, fontweight='bold', y=1.05)
    axes[4].set_xlabel('Genomic Position (bp)', fontsize=11)

    plt.tight_layout()
    output_pdf = "Figure7.pdf"
    output_png = "Figure7.png"
    plt.savefig(output_pdf, dpi=600, bbox_inches='tight')
    plt.savefig(output_png, dpi=600, bbox_inches='tight')
    print(f"\n[완료] Figure 7 타이틀 겹침 수정 및 번호 제거 완료! 저장 파일: {output_pdf}, {output_png}")

if __name__ == '__main__':
    run_comprehensive_analysis()
