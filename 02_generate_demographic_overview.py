# ==============================================================================
# File: 02_generate_demographic_overview.py
# Description: Demographic and Epidemiological Overview Script (Figure 1)
# ==============================================================================

import matplotlib
matplotlib.use('Agg') # 서버 환경 GUI 충돌 방지

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_figure1():
    meta_file = 'ppv_full_genomes_metadata_cleaned.csv'
    if not os.path.exists(meta_file):
        meta_file = 'ppv_2888_metadata.csv'
        
    if not os.path.exists(meta_file):
        print("[오류] 메타데이터 파일을 찾을 수 없습니다.")
        return

    df = pd.read_csv(meta_file)
    df.columns = [c.strip() for c in df.columns]

    # 시각화 스타일 설정
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    sns.set_theme(style="whitegrid")
    
    # 2x2 서브플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    # 폰트 크기 정의
    title_fs, label_fs, tick_fs = 15, 13, 11

    # A: Strain Distribution (진한 인디고/퍼플 톤 그라데이션 적용)
    if 'Strain' in df.columns:
        strain_counts = df['Strain'].value_counts().head(10)
        sns.barplot(x=strain_counts.values, y=strain_counts.index, ax=axes[0, 0], palette='Purples_r')
        axes[0, 0].set_title('Strain Distribution', fontsize=title_fs, fontweight='bold', pad=12)
        axes[0, 0].set_xlabel('Number of Isolates', fontsize=label_fs, fontweight='semibold')
        axes[0, 0].set_ylabel('Strain', fontsize=label_fs, fontweight='semibold')
        axes[0, 0].tick_params(labelsize=tick_fs)

    # B: Country Distribution (Top 10)
    if 'Country' in df.columns:
        country_counts = df['Country'].value_counts().head(10)
        sns.barplot(x=country_counts.values, y=country_counts.index, ax=axes[0, 1], palette='crest')
        axes[0, 1].set_title('Top 10 Countries', fontsize=title_fs, fontweight='bold', pad=12)
        axes[0, 1].set_xlabel('Number of Isolates', fontsize=label_fs, fontweight='semibold')
        axes[0, 1].set_ylabel('Country', fontsize=label_fs, fontweight='semibold')
        axes[0, 1].tick_params(labelsize=tick_fs)

    # C: Host Distribution (Top 10) - Prunus 이탤릭체 적용
    if 'Host' in df.columns:
        host_counts = df['Host'].value_counts().head(10)
        sns.barplot(x=host_counts.values, y=host_counts.index, ax=axes[1, 0], palette='mako')
        axes[1, 0].set_title('Top 10 Host Plants', fontsize=title_fs, fontweight='bold', pad=12)
        axes[1, 0].set_xlabel('Number of Isolates', fontsize=label_fs, fontweight='semibold')
        axes[1, 0].set_ylabel('Host', fontsize=label_fs, fontweight='semibold')
        axes[1, 0].tick_params(labelsize=tick_fs)
        
        for label in axes[1, 0].get_yticklabels():
            text_val = label.get_text()
            if text_val.startswith("Prunus"):
                label.set_fontstyle('italic')

    # D: Collection Year Trend
    if 'Year' in df.columns:
        df['Year_clean'] = pd.to_numeric(df['Year'], errors='coerce')
        year_counts = df['Year_clean'].dropna().astype(int).value_counts().sort_index()
        sns.lineplot(x=year_counts.index, y=year_counts.values, ax=axes[1, 1], marker='o', color='#1d4e89', linewidth=2.5, markersize=6)
        axes[1, 1].set_title('Collection Year Trend', fontsize=title_fs, fontweight='bold', pad=12)
        axes[1, 1].set_xlabel('Year', fontsize=label_fs, fontweight='semibold')
        axes[1, 1].set_ylabel('Number of Isolates', fontsize=label_fs, fontweight='semibold')
        axes[1, 1].tick_params(axis='x', rotation=45, labelsize=tick_fs)
        axes[1, 1].tick_params(axis='y', labelsize=tick_fs)

    plt.tight_layout()
    
    # 600 DPI 고해상도 PNG 및 벡터 PDF 포맷 동시 저장
    plt.savefig('Figure1_PPV_Metadata_Summary.png', dpi=600, bbox_inches='tight')
    plt.savefig('Figure1_PPV_Metadata_Summary.pdf', dpi=600, bbox_inches='tight')
    print("[완료] 학명 이탤릭체 및 타이틀 번호 제거가 적용되어 Figure 1 스크립트가 정비되었습니다.")

if __name__ == '__main__':
    generate_figure1()
