# PlantViroML-PPV

This repository contains official Python scripts, machine learning models, and supplementary datasets for the phylogenomic, recombination, and Explainable AI (XAI) analysis of **Plum pox virus (PPV)**, along with an interactive real-time genomic surveillance web dashboard.

## 📂 Repository Structure & Script List

The analysis and deployment pipeline comprises the following Python scripts:

| Script Name | Description |
| :--- | :--- |
| `01_integrate_rdp5_results.py` | Integrates and processes RDP5 recombination analysis outputs. |
| `02_generate_demographic_overview.py` | Generates demographic and epidemiological overview visualizations (Figure 1). |
| `03_generate_phylogenetic_embedding_correlation.py` | Analyzes and correlates phylogenetic trees with DNABERT-2 sequence embeddings (Figure 2). |
| `04_generate_recombination_umap_analysis.py` | Performs UMAP dimensionality reduction and clustering for viral recombination analysis (Figure 3). |
| `05_generate_xai_shap_analysis.py` | Executes Explainable AI (XAI) SHAP analysis for machine learning model interpretation (Figure 4). |
| `06_generate_phylodynamic_molecular_clock.py` | Conducts phylodynamic molecular clock and evolutionary rate estimations (Figure 5). |
| `07_generate_conservation_genome_map.py` | Generates genome-wide sequence conservation and mapping visualizations (Figure 7). |
| `08_generate_pan_ppv_diagnostics.py` | Evaluates pan-PPV diagnostic marker efficiency and conservation (Table S7). |
| `09_plantviroml_ppv_dashboard.py` | Runs the interactive PlantViroML-PPV genomic intelligence web dashboard. |
| `10_run_dnabert_shap.py` | Executes detailed DNABERT-2 embedding-based SHAP feature importance analysis. |
| `PlantViroML-PPV.py` | Core dashboard runner script for real-time genomic prediction. |

## 🧬 Built-in Example Genomic Datasets

The repository includes three built-in full-genome FASTA example files for testing and benchmarking the dashboard:

| File Name | Description & GenBank Accession | Strain / Isolate Type |
| :--- | :--- | :--- |
| `OK562672.1.fasta` | Plum pox virus isolate Y3, complete genome[cite: 1] | PPV-Y |
| `KF472134.1.fasta` | Plum pox virus strain Rec, complete genome[cite: 2] | PPV-Rec |
| `KP998124.1.fasta` | Plum pox virus strain PPV-D, complete genome[cite: 3] | PPV-D |

## 🚀 Web Dashboard Deployment (`09_plantviroml_ppv_dashboard.py`)
The repository includes an interactive Streamlit dashboard allowing users to analyze Plum Pox Virus genomes using three flexible input options:
1. **Use Example FASTA Files**: Quickly load and benchmark the built-in reference genomes (`OK562672.1`, `KF472134.1`, `KP998124.1`)[cite: 1, 2, 3].
2. **Upload FASTA File**: Upload a custom full-genome FASTA file (`.fasta`, `.fa`, `.txt`).
3. **Paste FASTA Sequence**: Directly paste FASTA format text into the input text box.

The dashboard instantly predicts:
* 🦠 **Strain Classification** (`ppv_Strain_rf_model.pkl`)
* 🌍 **Country Origin Classifier** (`ppv_Country_rf_model.pkl`)
* 🌳 **Host Adaptation** (`ppv_Host_rf_model.pkl`)
* 🧬 **Recombination Status** (`ppv_Recombinant_rf_model.pkl`)
* 🧪 **Mutation / Assay Status** (`ppv_Mutation_Status_rf_model.pkl`)

### Local Execution:
```bash
streamlit run 09_plantviroml_ppv_dashboard.py
