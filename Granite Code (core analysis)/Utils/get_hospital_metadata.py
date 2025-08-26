# Gets hospital metadata

from pathlib import Path
import pandas as pd
import numpy as np

# get hospitals in analysis from summaries .csv

project_2017_path = Path('/drives/56219-Linux/56219dua/Project2017')
export_path = project_2017_path / '4-Analysis/3-Analysis/AidanPaper/Export'
aidan_data_path = project_2017_path / '1-Data/AidanData'

hids_path = export_path / 'hospitalIDs.txt'
with open(hids_path, 'r') as hidsf:
    hids_str = hidsf.readlines()

hids = [int(hid.strip('\n')) for hid in hids_str]


print(f'Getting metadata for {len(hids)} hospitals')

assert len(hids) == len(list(set(hids))), f'hids has {len(hids)} items but only {len(list(set(hids)))} unique items'

# create a map from zip code to ruca code
rurality_path = aidan_data_path / 'rucaZipCut.csv'
with open(rurality_path, 'r') as rf:
    rflines = rf.readlines()

zip_to_ruca = {}
for line in rflines[1:]:
    line = line.strip("\n")
    z, rcode = line.split(",")
    zip_to_ruca[str(int(z))] = int(rcode)


# now create a map from hospital id to ruca code
hid_to_zip_path = aidan_data_path / 'Hospital_General_InformationCut.csv'
with open(hid_to_zip_path, 'r') as zf:
    zflines = zf.readlines()

hid_to_zip = {}
for line in zflines[1:]:
    line = line.strip("\n")
    hid, z = line.split(",")
    hid_to_zip[int(hid)] = z


failed_on_htoz = []
failed_on_ztor = []
hid_to_ruca = {}

for hid in hids:
    try:
        z = hid_to_zip[hid]
        try:
            r = zip_to_ruca[z]
            hid_to_ruca[hid] = r
        except KeyError: 
            failed_on_ztor.append(hid)
            print(f'no ruca code for zip = {z}')

    except KeyError:
        failed_on_htoz.append(hid)
        print(f'no zip for hid = {hid}')

print(f'{len(failed_on_htoz)} failed on converting hid to zip')
print(f'{len(failed_on_ztor)} failed on converting zip to ruca')

#print(np.array(list(hid_to_ruca.values())))
unique, counts = np.unique(np.array(list(hid_to_ruca.values())), return_counts=True)
ruca_counts = dict(zip(unique.tolist(), counts.tolist()))
print("Counts of Ruca Codes")
for code, count in ruca_counts.items():
    print(f'{code}: {count}')








    





# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# now we want to get the hospital type information.


with open(hids_path, 'r') as hidsf:
    hids_str = hidsf.readlines()

hids = [int(hid.strip('\n')) for hid in hids_str]

# fix Brigham and womens hid discrepency 
hids.remove(220110) # up to date one, but not in hospitalID_hospitalType.csv
hids.append(220162) # old one that is in the file we are trying to interact with

ruca_path = aidan_data_path / 'hospitalID_hospitalType.csv'

assert aidan_data_path == Path('/drives/56219-Linux/56219dua/Project2017/1-Data/AidanData/')

ruca_df = pd.read_csv(ruca_path)
ruca_df_in_analysis = ruca_df[ruca_df["hospitalID"].isin(hids)]


assert ruca_df_in_analysis['hospitalID'].nunique() == len(hids), f"found {ruca_df_in_analysis['hospitalID'].nunique()} unique hospital ids but expected {len(hids)} ."
assert set(ruca_df_in_analysis['hospitalID']) == set(hids)

print(ruca_df_in_analysis['HospitalType'].value_counts())







