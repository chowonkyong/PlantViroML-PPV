# PlantViroML-PPV

This repository contains official Python scripts, machine learning models, and supplementary datasets for the phylogenomic, recombination, and Explainable AI (XAI) analysis of **Plum pox virus (PPV)**.

## 📂 Repository Structure & Script List

The analysis pipeline comprises the following 10 core Python scripts:

| Script Name | Format | Size (Bytes) | Description |
| :--- | :--- | :--- | :--- |
| `01_integrate_rdp5_results.py` | PY | 4,994 | Integrates and processes RDP5 recombination analysis outputs. |
| `02_generate_demographic_overview.py` | PY | 4,244 | Generates demographic and epidemiological overview visualizations (Figure 1). |
| `03_generate_phylogenetic_embedding_correlation.py` | PY | 6,430 | Analyzes and correlates phylogenetic trees with DNABERT-2 sequence embeddings (Figure 2). |
| `04_generate_recombination_umap_analysis.py` | PY | 7,721 | Performs UMAP dimensionality reduction and clustering for viral recombination analysis (Figure 3). |
| `05_generate_xai_shap_analysis.py` | PY | 6,962 | Executes Explainable AI (XAI) SHAP analysis for machine learning model interpretation (Figure 4). |
| `06_generate_phylodynamic_molecular_clock.py` | PY | 6,761 | Conducts phylodynamic molecular clock and evolutionary rate estimations (Figure 5). |
| `07_generate_conservation_genome_map.py` | PY | 8,159 | Generates genome-wide sequence conservation and mapping visualizations (Figure 7). |
| `08_generate_pan_ppv_diagnostics.py` | PY | 9,378 | Evaluates pan-PPV diagnostic marker efficiency and conservation (Table S7). |
| `09_plantviroml_ppv_dashboard.py` | PY | 6,104 | Runs the interactive PlantViroML PPV analysis web dashboard. |
| `10_run_dnabert_shap.py` | PY | 6,781 | Executes detailed DNABERT-2 embedding-based SHAP feature importance analysis. |

## 🛠️ Usage
All scripts are designed to run within Python environments configured for the PlantViroML genomic pipeline. For details regarding supplementary data files, please refer to the project documentation.
