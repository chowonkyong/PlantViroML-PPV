# ==============================================================================
# File: 08_generate_pan_ppv_diagnostics.py
# Description: Pipeline to generate Supplementary Table S7 
#              (Pan-PPV Diagnostic Assay & CRISPR-Cas12a Pipeline)
# ==============================================================================

import os
import numpy as np
import pandas as pd
from Bio import AlignIO
from Bio.SeqUtils import MeltingTemp as mt
from Bio.Seq import Seq

def run_tables7_integrated_diagnostics():
    print("=== Table S7: Pan-PPV Diagnostic Assay & CRISPR-Cas12a Pipeline ===")
    
    alignment_file = 'ppv_full_genomes_aligned.fasta'
    if not os.path.exists(alignment_file):
        alignment_file = 'ppv_full_genomes_trimmed.fasta'
        
    alignment = AlignIO.read(alignment_file, "fasta")
    align_array = np.array([list(rec.seq.upper()) for rec in alignment])
    alignment_length = align_array.shape[1]
    num_seqs = len(alignment)
    
    print(f"Total sequences loaded: {num_seqs}, Alignment length: {alignment_length} bp")

    # ==========================================
    # Part 1: RT-qPCR Assay Design & Validation
    # ==========================================
    print("\n[Step 1] Running RT-qPCR Assay Validation...")
    fwd_start = 9520
    fwd_len = 20
    fwd_window = align_array[:, fwd_start:fwd_start+fwd_len]
    
    fwd_primer_list = []
    for col_idx in range(fwd_len):
        col = fwd_window[:, col_idx]
        valid_col = col[(col == 'A') | (col == 'C') | (col == 'G') | (col == 'T')]
        vals, counts = np.unique(valid_col, return_counts=True)
        fwd_primer_list.append(vals[np.argmax(counts)])
    fwd_primer = "".join(fwd_primer_list)

    best_rev_pos = -1
    min_rev_entropy = float('inf')
    search_start = fwd_start + fwd_len + 80
    search_end = fwd_start + fwd_len + 180
    
    for start in range(search_start, search_end - fwd_len + 1, 5):
        window_cols = align_array[:, start:start+fwd_len]
        if np.mean(window_cols == '-') > 0.05:
            continue
        col_entropies = []
        for col_idx in range(fwd_len):
            col = window_cols[:, col_idx]
            valid_col = col[(col == 'A') | (col == 'C') | (col == 'G') | (col == 'T')]
            if len(valid_col) == 0:
                col_entropies.append(1.0)
                continue
            _, counts = np.unique(valid_col, return_counts=True)
            freqs = counts / len(valid_col)
            entropy = -np.sum(freqs * np.log2(freqs + 1e-9))
            col_entropies.append(entropy)
        mean_ent = np.mean(col_entropies)
        if mean_ent < min_rev_entropy:
            min_rev_entropy = mean_ent
            best_rev_pos = start

    rev_window = align_array[:, best_rev_pos:best_rev_pos+fwd_len]
    rev_primer_list = []
    for col_idx in range(fwd_len):
        col = rev_window[:, col_idx]
        valid_col = col[(col == 'A') | (col == 'C') | (col == 'G') | (col == 'T')]
        vals, counts = np.unique(valid_col, return_counts=True)
        rev_primer_list.append(vals[np.argmax(counts)])
    
    rev_raw = "".join(rev_primer_list)
    rev_primer = str(Seq(rev_raw).reverse_complement())

    fwd_tm = mt.Tm_NN(Seq(fwd_primer))
    rev_tm = mt.Tm_NN(Seq(rev_primer))
    amplicon_size = best_rev_pos + fwd_len - fwd_start

    strain_results = []
    mismatch_counts_dict = {0: 0, 1: 0, 2: 0, '3+': 0}
    success_count = 0
    
    for seq_idx in range(num_seqs):
        rec_id = alignment[seq_idx].id.strip()
        fwd_target = "".join(fwd_window[seq_idx]).replace('-', '')
        rev_target_raw = "".join(rev_window[seq_idx]).replace('-', '')
        try:
            rev_target = str(Seq(rev_target_raw).reverse_complement())
        except:
            rev_target = "N" * fwd_len
            
        f_mismatch = sum(1 for a, b in zip(fwd_primer, fwd_target) if a != b)
        r_mismatch = sum(1 for a, b in zip(rev_primer, rev_target) if a != b)
        total_mismatch = f_mismatch + r_mismatch
        
        if total_mismatch in mismatch_counts_dict:
            mismatch_counts_dict[total_mismatch] += 1
        else:
            mismatch_counts_dict['3+'] += 1
            
        is_success = "Yes" if (f_mismatch <= 1 and r_mismatch <= 1) else "No"
        if is_success == "Yes":
            success_count += 1
            
        strain_results.append({
            "Strain_ID": rec_id,
            "Forward_Mismatches": f_mismatch,
            "Reverse_Mismatches": r_mismatch,
            "Total_Mismatches": total_mismatch,
            "Amplification_Success": is_success
        })

    # ==========================================
    # Part 2: Cas12a crRNA Internal Guide Design
    # ==========================================
    print("\n[Step 2] Scanning RT-qPCR Amplicon for Cas12a crRNA Guides...")
    spacer_len = 20
    target_start = fwd_start
    target_end = best_rev_pos + fwd_len
    
    cas12_candidates = []
    for start in range(target_start, target_end - 4 - spacer_len + 1):
        pam_cols = align_array[:, start:start+4]
        if np.any(pam_cols == '-'):
            continue
            
        is_ttv = True
        for c in range(3):
            col = pam_cols[:, c]
            vals, counts = np.unique(col, return_counts=True)
            if vals[np.argmax(counts)] != 'T':
                is_ttv = False
                break
        if not is_ttv:
            continue
            
        col4 = pam_cols[:, 3]
        vals4, counts4 = np.unique(col4, return_counts=True)
        v_base = vals4[np.argmax(counts4)]
        if v_base not in ['A', 'C', 'G']:
            continue
            
        pam_str = "TTT" + v_base
        spacer_start = start + 4
        spacer_window = align_array[:, spacer_start:spacer_start+spacer_len]
        
        if np.mean(spacer_window == '-') > 0.01:
            continue
            
        col_entropies = []
        for col_idx in range(spacer_len):
            col = spacer_window[:, col_idx]
            valid_col = col[(col == 'A') | (col == 'C') | (col == 'G') | (col == 'T')]
            if len(valid_col) == 0:
                col_entropies.append(1.0)
                continue
            _, counts = np.unique(valid_col, return_counts=True)
            freqs = counts / len(valid_col)
            entropy = -np.sum(freqs * np.log2(freqs + 1e-9))
            col_entropies.append(entropy)
            
        mean_ent = np.mean(col_entropies)
        cas12_candidates.append((spacer_start, pam_str, mean_ent, spacer_window))

    cas12_candidates.sort(key=lambda x: x[2])
    
    cas12_summary_data = []
    for rank, (pos, pam, ent, win_cols) in enumerate(cas12_candidates[:3], 1):
        guide_list = []
        for col_idx in range(spacer_len):
            col = win_cols[:, col_idx]
            valid_col = col[(col == 'A') | (col == 'C') | (col == 'G') | (col == 'T')]
            vals, counts = np.unique(valid_col, return_counts=True)
            guide_list.append(vals[np.argmax(counts)])
        guide_seq = "".join(guide_list)
        
        exact_matches = 0
        for seq_idx in range(num_seqs):
            strain_seq = "".join(win_cols[seq_idx]).replace('-', 'N')
            if strain_seq == guide_seq:
                exact_matches += 1
        coverage = (exact_matches / num_seqs) * 100

        cas12_summary_data.append({
            "Candidate_Rank": rank,
            "PAM_Sequence": pam,
            "Genomic_Position": f"{pos} ~ {pos + spacer_len}",
            "crRNA_Spacer_Sequence": guide_seq,
            "Mean_Entropy": round(ent, 5),
            "Exact_Match_Rate_Pct": round(coverage, 2)
        })

    # ==========================================
    # Part 3: Exporting Table S7 Results to CSV
    # ==========================================
    print("\n[Step 3] Exporting Table S7 results to CSV files...")
    
    df_details = pd.DataFrame(strain_results)
    df_details.to_csv("TableS7_Strain_Validation_Details.csv", index=False, encoding='utf-8-sig')

    summary_data = [
        {"Category": "RT-qPCR Assay", "Parameter": "Target Region", "Value": "CP / 3' UTR Junction"},
        {"Category": "RT-qPCR Assay", "Parameter": "Forward Primer (5'->3')", "Value": f"{fwd_primer} (Tm: {fwd_tm:.1f}°C)"},
        {"Category": "RT-qPCR Assay", "Parameter": "Reverse Primer (5'->3')", "Value": f"{rev_primer} (Tm: {rev_tm:.1f}°C)"},
        {"Category": "RT-qPCR Assay", "Parameter": "Target Amplicon Size", "Value": f"{amplicon_size} bp"},
        {"Category": "RT-qPCR Assay", "Parameter": "Total Evaluated Genomes", "Value": str(num_seqs)},
        {"Category": "RT-qPCR Assay", "Parameter": "Successful Amplifications", "Value": f"{success_count} ({(success_count/num_seqs)*100:.2f}%)"},
    ]
    df_assay_summary = pd.DataFrame(summary_data)
    df_assay_summary.to_csv("TableS7_RT_qPCR_Summary.csv", index=False, encoding='utf-8-sig')

    df_cas12 = pd.DataFrame(cas12_summary_data)
    df_cas12.to_csv("TableS7_Cas12a_crRNA_Candidates.csv", index=False, encoding='utf-8-sig')

    print("\n[완료] Table S7 통합 분석 결과가 다음의 파일들로 저장되었습니다:")
    print("  1. TableS7_RT_qPCR_Summary.csv")
    print("  2. TableS7_Cas12a_crRNA_Candidates.csv")
    print("  3. TableS7_Strain_Validation_Details.csv")

if __name__ == '__main__':
    run_tables7_integrated_diagnostics()
