import pandas as pd
import numpy as np
import json
from typing import Dict

def compute_test_comparison(df: pd.DataFrame) -> Dict:
    """Compute all test comparison variables from the summaries DataFrame."""
    comparisons = {}
    
    # 1. FE vs MW as a hospital classifier
    fe_positive = df['qmp'] < 0.05
    mw_positive = df['mwp'] < 0.05
    
    comparisons['fep_mwp'] = int(sum(fe_positive & mw_positive))
    comparisons['fep_mwn'] = int(sum(fe_positive & ~mw_positive))
    comparisons['fen_mwp'] = int(sum(~fe_positive & mw_positive))
    comparisons['fen_mwn'] = int(sum(~fe_positive & ~mw_positive))
    
    # 2. PCD(+/-) vs DBDFE(+/-) as a day classifier
    def parse_day_list(day_string: str) -> np.ndarray:
        return np.fromstring(day_string.strip('[]'), sep=' ', dtype=int)
    
    pcdp_dbdfep = 0
    pcdp_dbdfen = 0
    pcdn_dbdfep = 0
    pcdn_dbdfen = 0
    
    for _, row in df.iterrows():
        pmpcd_days = set(parse_day_list(row['pmpcd_sig_days']))
        wmpcd_days = set(parse_day_list(row['wmpcd_sig_days']))
        dbdfe_days = set(parse_day_list(row['dbdfe_sig_days']))
        
        # All PCD+ days (union of PMPCD and WMPCD)
        pcd_days = pmpcd_days | wmpcd_days
        
        # All possible days
        all_days = set(range(200))
        
        # Count days for each category
        pcdp_dbdfep += len(pcd_days & dbdfe_days)
        pcdp_dbdfen += len(pcd_days - dbdfe_days)
        pcdn_dbdfep += len(dbdfe_days - pcd_days)
        pcdn_dbdfen += len(all_days - (pcd_days | dbdfe_days))
    
    comparisons.update({
        'pcdp_dbdfep': pcdp_dbdfep,
        'pcdp_dbdfen': pcdp_dbdfen,
        'pcdn_dbdfep': pcdn_dbdfep,
        'pcdn_dbdfen': pcdn_dbdfen
    })
    
    # 3. FE vs PCD+ (PMPCD+, WMPCD+) as a day classifier
    comparisons['fep_pmpcdp'] = int(sum(fe_positive & (df['count_pmpcd_sig'] > 0)))
    comparisons['fep_wmpcdp'] = int(sum(fe_positive & (df['count_wmpcd_sig'] > 0)))
    comparisons['fep_pcdp'] = int(sum(fe_positive & ((df['count_pmpcd_sig'] + df['count_wmpcd_sig']) > 0)))
    comparisons['fep_pcdn'] = int(sum(fe_positive & ((df['count_pmpcd_sig'] + df['count_wmpcd_sig']) == 0)))
    
    comparisons['fen_pmpcdp'] = int(sum(~fe_positive & (df['count_pmpcd_sig'] > 0)))
    comparisons['fen_wmpcdp'] = int(sum(~fe_positive & (df['count_wmpcd_sig'] > 0)))
    comparisons['fen_pcdp'] = int(sum(~fe_positive & ((df['count_pmpcd_sig'] + df['count_wmpcd_sig']) > 0)))
    comparisons['fen_pcdn'] = int(sum(~fe_positive & ((df['count_pmpcd_sig'] + df['count_wmpcd_sig']) == 0)))
    
    # 4. MW vs ≥ n days PCD as a hospital classifier
    for n in [1, 5, 10]:
        pcd_ge_n = (df['count_pmpcd_sig'] + df['count_wmpcd_sig']) >= n
        
        comparisons[f'mwp_{n}_pcdp'] = int(sum(mw_positive & pcd_ge_n))
        comparisons[f'mwp_{n}_pcdn'] = int(sum(mw_positive & ~pcd_ge_n))
        comparisons[f'mwn_{n}_pcdp'] = int(sum(~mw_positive & pcd_ge_n))
        comparisons[f'mwn_{n}_pcdn'] = int(sum(~mw_positive & ~pcd_ge_n))
    
    return comparisons

def generate_test_comparison(path_to_summaries: str, path_to_output: str):
    """Generate test comparison JSON file from summaries."""
    df = pd.read_csv(path_to_summaries)
    comparisons = compute_test_comparison(df)
    
    with open(path_to_output, 'w') as f:
        json.dump(comparisons, f, cls=json.JSONEncoder)
    print(f'Saved test comparisons to {path_to_output}')

def main():
    generate_test_comparison(
        './1-Input/Test-Data/summaries.csv',
        './3-Output/Test-Data/test_comparison.json'
    )

if __name__ == '__main__':
    main()