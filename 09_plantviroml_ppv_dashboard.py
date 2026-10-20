# ==============================================================================
# File: 09_plantviroml_ppv_dashboard.py
# Description: PlantViroML-PPV Genomic Intelligence Dashboard (Streamlit)
#              - Supports Local Examples, File Upload, and Text Area Paste
# ==============================================================================

import os
import io
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
        model_filename = f'ppv_{name}_rf_model.pkl'
        if os.path.exists(model_filename):
            try:
                models[name] = joblib.load(model_filename)
            except Exception as e:
                print(f"Error loading {model_filename}: {e}")
    return models

models = load_trained_models()

# Load example sequences directly from local FASTA files or embedded string
@st.cache_data
def get_example_records():
    records = []
    
    # 1. Embedded MK387313.1 Sequence (기본 예제로 내장)
    mk387313_fasta = """>MK387313.1 Plum pox virus isolate W3174-02, complete genome
AAAATATAAAAACTCAACACAACCTACAAAATTTTATGCGATCAAACCAACTTCAAACTTTCAAAATCTT
AACAATCTTGCAGTGCTATTTGCGAAATACTTCTACATTTTACTTCATTCAGTTTTCCAGCAAACAACAG
TTAACGATGTCGACGATTGTGTTTGGCTCATTTACTTGTCATCTTGACGCAGCAATCCATCAGGATAATG
CAGATAGGTTGGCAAAGGCCTGGACTCGTCCAGAGAATCGCCAAGTCAGTAATGTGCATCTACTGTGCCG
AAGAACTGCAAAAAGTCTAATCGATACATATGAGAGYGCAACAGCTAGTGCGTGGAAAGGGTTAGAGGAG
AAGCTCCAACCAATGTTTGCTAAACGAGAATTTAGCAAAACAGTCACAAAGAGGAAGGGGCTTCGTTGTT
TTAAGACAAGTTCTGAGCATTTCATCGAAAAGAAGCTCAAGCAGCTATACAATGAGGAACGTGAACGTTT
TCAGTTCACCAATGGTCCAGATAGCGTGATCACACGCATTGGATCAGACAATGAGGCGACACCAGTGTGG
GTGCCTTTTCCCCAAATTCAGAGGAAACCGCTCTGTTCCACACACTCCATGAAGAAGAAGGTGGTTTTTG
CGAAAGTGAAAATGTCTGAGAAAACCCTACAGCTGTTTATGCGGAAGATAGCCGCTAATGCAAAAGCAAA
TGGGCAGAAAGTTGAAATCATAGGAAAGAAACATGTGGTAGGAAACTACATAACCAAAAGCAAACTGACG
TACTTTCGCACACACGTGCGACATTTGGATGGGTCGAAACCACGGTACGATATTGTCCTTGATGAGGCAA
CGAAGAAAATTTTGCAACTCTTTTCAGAAACGAGTGGTTTCAAACACCTCCATAATAAAGGTGAGATAAC
TCCAGGAATGAGTGGGTTTGTTATTAATCCGATCAATATATCGGATCCAATGCAAATTCACGATACCGAT
TTGTTCATAGTGCGTGGCAAACGTAACTCTACACTTGTTGATTCGCGATGCCCAATCTCGAAAGAGCACA
GTAATGAGTTAATTCACTATGCAGATCCAGGAAAGAAGTTTTGGGAGGGCTTCACAAATTCATTTGCGCA
ATGCAAGTTACGAGAAACTGATCATCAGTGCACCTCTGACTTGGACGTTAAGGAGTGTGGGTATGTAGCA
GCTCTCGTTTGTCAAGCCATCATTCCTTGTGGAAAAATCACATGCTTGCAGT
