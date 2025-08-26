import pandas as pd
from pathlib import Path
from get_example_hospital_results import generate_example_hospital_results
from tqdm import tqdm

def filter_hospitals_by_pcd_sig_days(summaries_path: str) -> pd.DataFrame:
    df = pd.read_csv(summaries_path)

    hids = df['hospitalID'].tolist()

    pm_ex = []
    wm_ex = []
    for hid in tqdm(hids):
        try:
            example_results = generate_example_hospital_results(hid)

            # get the number of exportable wmpcdp days
            # get the number of exportable pmpcdp days

            ex_wmpcdp_days = sum([1 if val > 0.95 else 0 for val in example_results['wmpcd']])
            ex_pmpcdp_days = sum([1 if val > 0.95 else 0 for val in example_results['pmpcd']])

            if ex_wmpcdp_days > 0:
                wm_ex.append((hid, ex_wmpcdp_days))
            elif ex_pmpcdp_days > 0:
                pm_ex.append((hid, ex_pmpcdp_days))

        except (AssertionError, IndexError):
            print(f'skipping {hid}')

    print("done identifying hospitals")
    pm_ex.sort(key=lambda x: x[1])
    wm_ex.sort(key=lambda x: x[1])

    print("PMEX")
    print(pm_ex)
    print("WMEX")
    print(wm_ex)

#    df['count_pcd_sig'] = df['count_wmpcd_sig'] + df['count_pmpcd_sig']
#
#    filtered_df = df[
#            (df['qmp'] > 0.05) &
#            (df['count_pcd_sig'] >= threshold)
#            ]
#
#    filtered_df.loc[:,'min_utilization'] = filtered_df[['u_p', 'u_w']].min(axis=1)
#
#    result_df = filtered_df.sort_values('min_utilization', ascending=False)
#
#    return result_df


def main():
    save_path = Path('/drives/drive1/56219dua/Project2017/4-Analysis/3-Analysis/AidanPaper')
    
    result_df = filter_hospitals_by_pcd_sig_days(save_path / 'summaries.csv')
    print(result_df.head(10))

if __name__ == '__main__':
    main()
