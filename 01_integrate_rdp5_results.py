# ==============================================================================
# File: 01_integrate_rdp5_results.py
# Description: RDP5 Recombination Results Integration Script
# ==============================================================================

import pandas as pd
import csv
import os

def integrate_rdp5():
    mapping_file = "full_seq_mapping_table.csv"
    rdp_file = "ppv-188-rdp5.csv"
    output_file = "integrated_rdp5_full_results.csv"
    
    if not os.path.exists(mapping_file) or not os.path.exists(rdp_file):
        print(f"[오류] 필수 입력 파일('{mapping_file}' 또는 '{rdp_file}')을 찾을 수 없습니다.")
        return

    # 1. 맵핑 파일 로드 (Seq_ID -> 원본 Accession 리스트)
    mapping_df = pd.read_csv(mapping_file)
    seq_to_acc = mapping_df.groupby('RDP5_Seq_ID')['Original_Accession'].apply(list).to_dict()

    # Seq 문자열 정리에 쓰일 헬퍼 함수
    def clean_seq_id(s):
        s = s.replace('^', '').replace('~', '').replace('*', '').strip()
        if s.startswith('Seq_'):
            return s
        return s

    # 2. RDP5 결과 파일 파싱
    events_dict = {}
    with open(rdp_file, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        header_found = False
        current_event_num = None
        
        for row in reader:
            if len(row) > 0 and 'Recombination Event Number' in row[0]:
                header_found = True
                continue
            if not header_found or len(row) == 0:
                continue
            
            col0 = row[0].strip()
            if col0:
                ev_num = col0.replace('~', '').replace('*', '').strip()
                if ev_num.isdigit():
                    current_event_num = ev_num
                    if current_event_num not in events_dict:
                        events_dict[current_event_num] = {
                            'Event_Num': current_event_num,
                            'Begin': row[2].strip() if len(row) > 2 else '',
                            'End': row[3].strip() if len(row) > 3 else '',
                            'Recombinants': set(),
                            'Minor_Parents': set(),
                            'Major_Parents': set()
                        }
            
            if current_event_num and current_event_num in events_dict:
                ev = events_dict[current_event_num]
                if len(row) > 2 and row[2].strip() and not ev['Begin']:
                    ev['Begin'] = row[2].strip()
                if len(row) > 3 and row[3].strip() and not ev['End']:
                    ev['End'] = row[3].strip()
                    
                recomb = row[8].strip() if len(row) > 8 else ''
                minor = row[9].strip() if len(row) > 9 else ''
                major = row[10].strip() if len(row) > 10 else ''
                
                if recomb: ev['Recombinants'].add(recomb)
                if minor: ev['Minor_Parents'].add(minor)
                if major: ev['Major_Parents'].add(major)

    # 3. Seq_ID를 653개 전체 원본 Accession 번호로 확장
    integrated_rows = []
    for ev_num, data in events_dict.items():
        recomb_accs = []
        for r in data['Recombinants']:
            clean_r = clean_seq_id(r)
            if clean_r in seq_to_acc:
                recomb_accs.extend(seq_to_acc[clean_r])
            else:
                recomb_accs.append(r)
                
        minor_accs = []
        for m in data['Minor_Parents']:
            clean_m = clean_seq_id(m)
            if clean_m in seq_to_acc:
                minor_accs.extend(seq_to_acc[clean_m])
            else:
                minor_accs.append(m)
                
        major_accs = []
        for maj in data['Major_Parents']:
            clean_maj = clean_seq_id(maj)
            if clean_maj in seq_to_acc:
                major_accs.extend(seq_to_acc[clean_maj])
            else:
                major_accs.append(maj)
                
        integrated_rows.append({
            'Event_Num': ev_num,
            'Breakpoint_Begin': data['Begin'],
            'Breakpoint_End': data['End'],
            'Recombinant_Clusters': ", ".join(sorted(data['Recombinants'])),
            'Recombinant_Accessions': ", ".join(sorted(list(set(recomb_accs)))),
            'Minor_Parent_Clusters': ", ".join(sorted(data['Minor_Parents'])),
            'Minor_Parent_Accessions': ", ".join(sorted(list(set(minor_accs)))),
            'Major_Parent_Clusters': ", ".join(sorted(data['Major_Parents'])),
            'Major_Parent_Accessions': ", ".join(sorted(list(set(major_accs))))
        })

    out_df = pd.DataFrame(integrated_rows)
    out_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"[완료] 총 {len(out_df)}개의 재조합 이벤트가 653개 전체 원본 서열 기준으로 통합되어 '{output_file}'에 저장되었습니다.")

if __name__ == "__main__":
    integrate_rdp5()
