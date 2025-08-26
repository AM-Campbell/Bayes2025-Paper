import pandas as pd
import numpy as np
from typing import Dict, List

# Generate example summaries.csv
def create_example_summaries(output_path: str, num_hospitals: int = 10):
    """Create an example summaries.csv file with realistic looking data."""
    data = []
    for i in range(num_hospitals):
        n_w = np.random.randint(100, 500)
        n_p = np.random.randint(50, 300)
        u_w = np.random.randint(0, n_w)
        u_p = np.random.randint(0, n_p)
        
        # Generate random significant days
        dbdfe_days = sorted(np.random.choice(200, size=np.random.randint(0, 30), replace=False))

        # First generate total number of PCD days we want
        total_pcd_days = np.random.randint(0, 50)  # Increased range since we'll split these
        
        # Randomly split these days between PMPCD and WMPCD
        all_days = np.random.choice(200, size=total_pcd_days, replace=False)
        split_point = np.random.randint(0, len(all_days) + 1)
        
        # Assign days to PMPCD and WMPCD ensuring no overlap
        pmpcd_days = sorted(all_days[:split_point])
        wmpcd_days = sorted(all_days[split_point:])
        
        
        # Calculate contingency table counts for FE vs PCD
        fep_bp = len(set(dbdfe_days) & set(pmpcd_days + wmpcd_days))
        fep_bn = len(set(dbdfe_days) - set(pmpcd_days + wmpcd_days))
        fen_bp = len(set(pmpcd_days + wmpcd_days) - set(dbdfe_days))
        fen_bn = 200 - fep_bp - fep_bn - fen_bp
        
        data.append({
            'hospitalID': i,
            'function': 'HospiceTreat',
            'n_w': n_w,
            'n_p': n_p,
            'u_w': u_w,
            'u_p': u_p,
            'count_dbdfe_sig': len(dbdfe_days),
            'dbdfe_sig_days': f"[{' '.join(map(str, dbdfe_days))}]",
            'count_wmpcd_sig': len(wmpcd_days),
            'wmpcd_sig_days': f"[{' '.join(map(str, wmpcd_days))}]",
            'count_pmpcd_sig': len(pmpcd_days),
            'pmpcd_sig_days': f"[{' '.join(map(str, pmpcd_days))}]",
            'mwp': np.random.uniform(0, 1),
            'qmp': np.random.uniform(0, 1),
            'fep_bp': fep_bp,
            'fep_bn': fep_bn,
            'fen_bp': fen_bp,
            'fen_bn': fen_bn
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    return df

def main():
    path = './1-Input/Test-Data/summaries.csv'
    df = create_example_summaries(path)
    print(f'Saved example summaries to {path}')

if __name__ == '__main__':
    main()


