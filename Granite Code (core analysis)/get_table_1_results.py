from pathlib import Path
import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.stats import mannwhitneyu, chi2_contingency
from dateutil.relativedelta import relativedelta


def main():
    PCD_THRESH = 0.05
    LOOKBACK_DAYS = 200
    summaries_dir = Path('/drives/drive1/56219dua/Project2017/4-Analysis/3-Analysis/AidanPaper')
    summaries_filename = f"summaries_{PCD_THRESH}_LBD_{LOOKBACK_DAYS}.csv"
    summaries_path = summaries_dir / summaries_filename

    pt_age_and_sex_file_path = Path("/drives/drive1/56219dua/Project2017/1-Data/2017/")
    filename = "eol17cacohortl6m_wHospitals.csv"

    pt_to_icd_path = Path('/drives/drive1/56219dua/Project2017/4-Analysis/3-Analysis/AidanPaper/BENE_ID_to_ICD.csv')
    pt_icd_df = pd.read_csv(pt_to_icd_path)
    print(pt_icd_df.head(10))

    icd_to_ccs_path = Path("/drives/56219-Linux/56219dua/Project2017/1-Data/AidanData/ccs_dx_icd10cm_2019.csv")
    icd_ccs_df = pd.read_csv(icd_to_ccs_path)
    icd_ccs_df = icd_ccs_df.apply(lambda x: x.str.replace("'", "") if x.dtype == "object" else x)
    icd_ccs_df.columns = icd_ccs_df.columns.str.strip("'")
    print(icd_ccs_df.head(10))
    print(icd_ccs_df.columns)

    most_common_ccs_save_path = Path("/drives/drive1/56219dua/Project2017/4-Analysis/3-Analysis/AidanPaper/Export/most_common_cancer_ccs.csv")
    table_1_age_sex_path = Path("/drives/drive1/56219dua/Project2017/4-Analysis/3-Analysis/AidanPaper/Export/Table1Info.csv")


    pt_df = pd.read_csv(pt_age_and_sex_file_path/filename)

    sumdf = pd.read_csv(summaries_path)
    
    hids =  sumdf['hospitalID'].tolist()

    pt_df = pt_df[pt_df['hospital'].isin(hids)]
    assert len(pt_df) == 100480 # the number of pt in our study

    # Add "Age" to DF 
    def calculate_age(dob, dod):
        dob = datetime.strptime(dob, '%Y-%m-%d')
        dod = datetime.strptime(dod, '%Y-%m-%d')
        return relativedelta(dod, dob).years

    pt_df['Age'] = pt_df.apply(lambda row: calculate_age(row['SDOB'], row['SDOD']), axis=1)
    pt_df['White'] = pt_df['RTI_RACE'] == 1
    print(pt_df.head())

    white_pts = pt_df[pt_df['RTI_RACE'] == 1]
    poc_pts = pt_df[pt_df['RTI_RACE'] != 1]

    with open(table_1_age_sex_path, 'w') as t1f:

        print('\n=== DEMOGRAPHICS ===', file=t1f)
        print(f'Total Patients: {len(pt_df)}', file=t1f)
        print(f'White Patients: {len(white_pts)}', file=t1f)
        print(f'POC Patients: {len(poc_pts)}', file=t1f)

        # 1. Age Analysis
        white_ages = white_pts['Age'].dropna()
        poc_ages = poc_pts['Age'].dropna()

        print("\n--- AGE ---", file=t1f)

        print("White Patients - Age:", file=t1f)
        print(f"    Median [Q1 - Q3]: {white_ages.median():.1f} [{white_ages.quantile(0.25):.1f} - {white_ages.quantile(0.75):.1f}]", file=t1f)


        print("POC Patients - Age:", file=t1f)
        print(f"    Median [Q1 - Q3]: {poc_ages.median():.1f} [{poc_ages.quantile(0.25):.1f} - {poc_ages.quantile(0.75):.1f}]", file=t1f)

        mw_stat, mw_p = mannwhitneyu(white_ages, poc_ages, alternative='two-sided')
        print(f"\nMann-Whitney U test: U={mw_stat}, p={mw_p:.4f}", file=t1f)      

        # Sex Analysis
        print("\n--- SEX ---", file=t1f)

        sex_summary = []
        for race_val, race_label in [(1, "White"), (0, "POC")]:
            if race_val == 1:
                subset = pt_df[pt_df['RTI_RACE'] == 1]
            else:
                subset = pt_df[pt_df['RTI_RACE'] != 1]

            total = len(subset)
            male = len(subset[subset['SEX'] == 1])
            female = len(subset[subset['SEX'] == 2])
            assert male + female == total

            print(f"{race_label} patients (n = {total}):", file=t1f)
            print(f"    Male: {male} ({male / total * 100:.1f}%)", file=t1f)
            print(f"    Female: {female} ({female / total * 100:.1f}%)", file=t1f)

            sex_summary.append([male, female])

        contingency_array = np.array(sex_summary)
        chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_array)
        print(f"\nChi-square test: stat={chi2_stat:.4f}, p={chi2_p:.4f}, dof={dof}", file=t1f)
    

    # Get Pt. ID's, Hospitals, and ICD-10-CM Codes
    pt_df['race_category'] = pt_df['RTI_RACE'].apply(lambda x: 'White' if x == 1 else 'POC')
    
    study_diagnoses = pt_df[['BENE_ID', 'race_category']].merge(
            pt_icd_df,
            on='BENE_ID',
            how='inner'
            )

    study_diagnoses_ccs = study_diagnoses.merge(
            icd_ccs_df,
            left_on='icd_10_code',
            right_on='ICD-10-CM CODE',
            how='left'
            )

    # Filter to only include cancer CCS Categories
    print(study_diagnoses_ccs['CCS CATEGORY'].dtype)
    print(study_diagnoses_ccs['CCS CATEGORY'].head(30))

    study_diagnoses_ccs['CCS CATEGORY'] = pd.to_numeric(study_diagnoses_ccs['CCS CATEGORY'], errors='coerce')

    ccs_range = [11,44]
    study_diagnoses_ccs = study_diagnoses_ccs[study_diagnoses_ccs['CCS CATEGORY'].between(ccs_range[0], ccs_range[1])]

    unmapped = study_diagnoses_ccs['CCS CATEGORY DESCRIPTION'].isna().sum()
    print(f"Number of unmapped ICD codes: {unmapped}")

    # unmapped_codes = study_diagnoses_ccs[study_diagnoses_ccs['CCS CATEGORY DESCRIPTION'].isna()]
    # print("some unmapped codes")
    # print(unmapped_codes['icd_10_code'].value_counts().head(20))


    patient_ccs = study_diagnoses_ccs.groupby(['BENE_ID', 'race_category', 'CCS CATEGORY DESCRIPTION']).size().reset_index(name='count')

    race_totals = pt_df['race_category'].value_counts()

    ccs_summary = []
    for ccs_category in patient_ccs['CCS CATEGORY DESCRIPTION'].unique():
        if pd.isna(ccs_category):
            continue

        # get unique patients with this dx by race
        patients_with_dx = patient_ccs[patient_ccs['CCS CATEGORY DESCRIPTION'] == ccs_category]
        
        white_count = patients_with_dx[patients_with_dx['race_category'] == 'White']['BENE_ID'].nunique()
        poc_count = patients_with_dx[patients_with_dx['race_category'] == 'POC']['BENE_ID'].nunique()

        white_pct = (white_count / race_totals['White']) * 100
        poc_pct = (poc_count / race_totals['POC']) * 100

        white_without_dx = race_totals['White'] - white_count
        poc_without_dx = race_totals['POC'] - poc_count

        cont_table = [
                [white_count, white_without_dx],
                [poc_count, poc_without_dx]
                ]

        chi2, p_val, dof, expected = chi2_contingency(cont_table)


        ccs_summary.append({
            "CCS_Category": ccs_category,
            'White_N': white_count,
            'White_Pct': white_pct,
            'POC_N': poc_count,
            'POC_Pct': poc_pct,
            'Total_N': white_count + poc_count,
            'chi2': chi2,
            'chi2_p': p_val
        })

    ccs_results_df = pd.DataFrame(ccs_summary)
    ccs_results_df = ccs_results_df.sort_values('Total_N', ascending=False)

    print(f'w: {race_totals["White"]}')
    print(f'p: {race_totals["POC"]}')
    print("--- Top Diagnoses by Prevalence ---")
    print(ccs_results_df.head(10))
    ccs_results_df.to_csv(most_common_ccs_save_path, index=True, sep=';')







if __name__ == "__main__":
    main()
    


