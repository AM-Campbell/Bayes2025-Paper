import pandas as pd
import glob
from pathlib import Path
from tqdm import tqdm

def create_icd_mapping(search_dir, out_file):

    csv_files = list(Path(search_dir).rglob('*.csv')) 
    print(f'Found {len(csv_files)} CSV files to process')

    total_records = 0
    processed_files = 0

    with open(out_file, 'w') as f_out:
        writer = None

        for fp in tqdm(csv_files, desc="Processing files", unit='file'):
            try:

                df = pd.read_csv(fp, dtype={'BENE_ID': str}, low_memory=False)
                icd_cols = [c for c in df.columns if c.startswith('ICD_DGNS')]
                if 'BENE_ID' not in df.columns or not icd_cols:
                    continue

                tidy = (
                        df.melt(id_vars='BENE_ID', value_vars=icd_cols, value_name='icd_10_code')
                        .dropna(subset=['icd_10_code'])
                        .loc[lambda d: d.icd_10_code.ne('')]
                        .drop_duplicates()
                        )

                record_count = len(tidy)
                total_records += record_count
                processed_files += 1

                if writer is None:
                    writer = tidy.to_csv(f_out, index=False, header=True, mode='a')
                    writer = True
                else:
                    writer = tidy.to_csv(f_out, index=False, header=False, mode='a')

                tqdm.write(f"\u2713 {fp.name}: {record_count:,} patient:ICD pairs extracted")

            except Exception as e:
                tqdm.write(f"Error processing {fp.name}: {str(e)}")


def main():
    search_dir = Path('/drives/56219-Linux/56219dua/Project2017/1-Data/2017/')
    output_file_path = Path('/drives/56219-Linux/56219dua/Project2017/4-Analysis/3-Analysis/AidanPaper/BENE_ID_to_ICD.csv')

    create_icd_mapping(search_dir, output_file_path)



if __name__ == "__main__":
    main()
