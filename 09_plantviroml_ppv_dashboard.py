# ==============================================================================
# File: 09_plantviroml_ppv_dashboard.py
# Description: PlantViroML-PPV Genomic Intelligence Dashboard (Streamlit)
#              - Reads example sequences directly from local FASTA files with path tolerance
# ==============================================================================

import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from Bio import SeqIO

# Streamlit page configuration
st.set_page_config(
    page_title="PlantViroML-PPV Genomic Surveillance",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load trained models with caching
@st.cache_resource
def load_trained_models():
    models = {}
    target_names = ['Strain', 'Country', 'Host', 'Recombinant', 'Mutation_Status']
    for name in target_names:
        model_filename = f'plantviroml_ppv_{name}_rf_model.pkl'
        if os.path.exists(model_filename):
            try:
                models[name] = joblib.load(model_filename)
            except Exception as e:
                print(f"Error loading {model_filename}: {e}")
    return models

models = load_trained_models()

# Load example sequences directly from local FASTA files (Path-tolerant version)
@st.cache_data
def get_example_records():
    example_filenames = [
        'OK562672.1.fasta', 
        'KF472134.1.fasta', 
        'KP998124.1.fasta'
    ]
    records = []
    # Search in current directory and common subdirectories (e.g., data)
    search_dirs = ['.', 'data', 'fasta', 'genomes']
    
    for filename in example_filenames:
        file_found = False
        for d in search_dirs:
            full_path = os.path.join(d, filename)
            if os.path.exists(full_path):
                try:
                    for record in SeqIO.parse(full_path, "fasta"):
                        records.append(record)
                    file_found = True
                    break
                except Exception as e:
                    print(f"Error reading {full_path}: {e}")
        if not file_found:
            print(f"Warning: {filename} could not be found.")
            
    return records

# Sidebar configuration
st.sidebar.title("PlantViroML-PPV")
st.sidebar.markdown(
    "Plum Pox Virus AI Surveillance Platform\n\n"
    "Active AI Models:\n"
    "- Strain Classifier\n"
    "- Country Origin Classifier\n"
    "- Host Adaptation Classifier\n"
    "- Recombination Classifier\n"
    "- Mutation / Assay Status\n\n"
    "Diagnostic Module:\n"
    "- Pan-PPV RT-qPCR & Cas12a In Silico Check"
)

st.sidebar.markdown("---")
input_mode = st.sidebar.radio("Select Input Source:", ["Use Local Example FASTA Files", "Upload Custom FASTA"])

records = []
if input_mode == "Use Local Example FASTA Files":
    records = get_example_records()
    if len(records) > 0:
        st.sidebar.success(f"Successfully loaded {len(records)} local example genomes[cite: 1, 2, 3].")
    else:
        st.sidebar.warning("Example FASTA files not found. Please ensure OK562672.1.fasta, KF472134.1.fasta, and KP998124.1.fasta are in the repository root[cite: 1, 2, 3].")
else:
    uploaded_file = st.sidebar.file_uploader("Upload PPV Full-Genome FASTA", type=["fasta", "fa", "txt"])
    if uploaded_file is not None:
        with open("temp_uploaded.fasta", "wb") as f:
            f.write(uploaded_file.getbuffer())
        records = list(SeqIO.parse("temp_uploaded.fasta", "fasta"))

# Main screen title and description
st.title("PlantViroML-PPV: AI-Driven Genomic Intelligence Dashboard")
st.markdown("Analyze Plum Pox Virus genomes using local example files or custom FASTA uploads to predict strain, geographic origin, host adaptation, recombination status, and diagnostic performance[cite: 1, 2, 3, 4].")

if len(records) > 0:
    selected_seq_id = st.selectbox("Select Isolate for Analysis:", [rec.id for rec in records])
    selected_record = next(rec for rec in records if rec.id == selected_seq_id)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Sequence Metadata")
        st.text(f"ID: {selected_record.id}")
        st.text(f"Description: {selected_record.description}")
        st.text(f"Length: {len(selected_record.seq)} bp")
        
        seq_str = str(selected_record.seq).upper()
        gc_content = (seq_str.count('G') + seq_str.count('C')) / len(seq_str) * 100
        st.text(f"GC Content: {gc_content:.2f}%")
        
        if st.button("Run PlantViroML-PPV Prediction", type="primary"):
            st.session_state['analyzed'] = True
    
    with col2:
        st.subheader("Sequence Preview")
        st.code(seq_str[:150] + "...", language="text")

    # Prediction results dashboard
    if st.session_state.get('analyzed', False):
        st.markdown("---")
        st.header("PlantViroML-PPV Real-Time Prediction Results")
        
        if not models:
            st.warning("Trained model files (.pkl) not found. Displaying simulation results based on sample metadata.")
            class DummyModel:
                def __init__(self, classes):
                    self.classes_ = classes
                def predict(self, x):
                    return [self.classes_[0]]
                def predict_proba(self, x):
                    p = np.array([0.85, 0.10, 0.05])
                    return p.reshape(1, -1)
            
            models = {
                'Strain': DummyModel(['PPV-Y', 'PPV-Rec', 'PPV-D']),
                'Country': DummyModel(['France', 'Spain', 'Italy']),
                'Host': DummyModel(['Prunus persica', 'Prunus domestica', 'Armeniaca']),
                'Recombinant': DummyModel(['Non-Recombinant', 'Recombinant']),
                'Mutation_Status': DummyModel(['Wild-type', 'Mutant'])
            }
        
        np.random.seed(hash(selected_record.id) % 2**32
