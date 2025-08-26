# generates json file for hospital scatter plot

import pandas as pd
from pathlib import Path
import json
import numpy as np

def get_scatter_data(df):
    results = []
    for _, row in df.iterrows():
        wm_count = row['count_wmpcd_sig']
        pm_count = row['count_pmpcd_sig']

        if wm_count > 0 and pm_count > 0: 
                raise ValueError(f"Hospital {row['hospitalID']} has both WMPCD and PMPCD days")

        if wm_count > 0:
            pcd_type = 'WM'
            ar_col = 'wmpcd_sig_days'
            num_pcdp_days = wm_count
        elif pm_count > 0:
            pcd_type = 'PM'
            ar_col = 'pmpcd_sig_days'
            num_pcdp_days = pm_count
        else:
            continue
        
        pcd_days_list = [int(x) for x in row[ar_col].strip('[]').split()]

        assert len(pcd_days_list) > 0, 'pcd_days_list should not be empty'

        avg_days = float(np.mean(pcd_days_list))

        hospital_data = {
                'hospitalID': int(row['hospitalID']),
                'num_pcdp_days': num_pcdp_days,
                'avg_days_before_death': float(avg_days),
                'pcd_type': pcd_type,
                'fep': bool(row['qmp'] < 0.05)
                }

        results.append(hospital_data)

    return results


def get_hospitals_list(df):
    return df['hospitalID'].tolist()

def main():
    save_path = Path('/drives/drive1/56219dua/Project2017/4-Analysis/3-Analysis/AidanPaper')
    df = pd.read_csv(save_path / 'summaries_0.05_LBD_200.csv')
    hospitals_list = get_hospitals_list(df)
    
    hospitals_list_path = save_path / 'Export' / 'hospitalIDs.txt'
    with open(hospitals_list_path, 'w') as hf:
            hf.writelines((f'{hospitalID}\n' for hospitalID in hospitals_list))

    print(f'done writing hospital IDs to {hospitals_list_path}')

    scatter_list = get_scatter_data(df)

    scatter_save_path = save_path / 'Export' / 'pcdp_hospitals_scatter_plot_data.json'

    with open(scatter_save_path, 'w') as jf:
        json.dump(scatter_list, jf, indent=2)

    print(f'done writing PCD(+) hospital data to {scatter_save_path}')

if __name__ == '__main__':
    main()



