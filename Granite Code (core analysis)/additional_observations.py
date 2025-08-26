import ast
import json
from typing import Dict, List
import pandas as pd
import numpy as np
from pathlib import Path


def compute_observations(df: pd.DataFrame) -> Dict[str, float]:
    observations = {}

    # Basic counts (correct)
    observations["num_patients"] = df["n_w"].sum() + df["n_p"].sum()
    observations["num_hospitals"] = len(df)
    
    # Check that FE Valence is PCD Valence
    fdf = df[(df["qmp"] < 0.05) & (df["count_wmpcd_sig"] > 0)]
    #print(f'len fdf is {len(fdf)}')
    #print("WM FE WMPCD")
    #print(fdf[fdf["u_w"]/fdf["n_w"] > fdf["u_p"]/fdf["n_p"]])
    prop1 = len(fdf[fdf["u_w"]/fdf["n_w"] < fdf["u_p"]/fdf["n_p"]]) == 0
    

    fdf = df[(df["qmp"] < 0.05) & (df["count_pmpcd_sig"] > 0)]
    #print(f'len fdf is {len(fdf)}')
    #print("PM FE PMPCD")
    #print(fdf[fdf["u_w"]/fdf["n_w"] < fdf["u_p"]/fdf["n_p"]])
    prop2 = len(fdf[fdf["u_w"]/fdf["n_w"] > fdf["u_p"]/fdf["n_p"]]) == 0

    print(prop1, bool(prop1))
    print(prop2, bool(prop2))
    observations['fevalence_is_pcdvalence'] = prop1 and prop2
    print(f'observations["fevalence_is_pcdvalence"] = {prop1 and prop2}')

    # Count the number of WM FE hospitals
    observations["count_wmfe"] = len(df[(df["qmp"] < 0.05) & (df["u_w"]/df["n_w"] > df["u_p"]/df["n_p"])])
    observations["count_pmfe"] = len(df[(df["qmp"] < 0.05) & (df["u_w"]/df["n_w"] < df["u_p"]/df["n_p"])])

    print(f'observations["count_wmfe"] = {observations["count_wmfe"]}')
    print(f'observations["count_pmfe"] = {observations["count_pmfe"]}')

    observations["percent_wmfe"] =  observations["count_wmfe"] / observations['num_hospitals']
    observations["percent_pmfe"] =  observations["count_pmfe"] / observations['num_hospitals']
    
    print(f'observations["percent_wmfe"] = {observations["percent_wmfe"]}')
    print(f'observations["percent_pmfe"] = {observations["percent_pmfe"]}')


    def parse_day_list(day_string: str) -> np.ndarray:
        # Remove brackets and convert to numpy array
        return np.fromstring(day_string.strip("[]"), sep=" ", dtype=int)

    return observations


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def generate_observations(path_to_summaries, path_to_output) -> Dict:
    df = pd.read_csv(path_to_summaries)
    observations = compute_observations(df)
    with open(path_to_output, "w") as f:
        json.dump(observations, f, cls=NumpyEncoder)
    print(f"Saved observations to {path_to_output}")
    return observations


def main():
    save_path = Path(
        "/drives/drive1/56219dua/Project2017/4-Analysis/3-Analysis/AidanPaper"
    )
    summaries_path = save_path / "summaries_0.05_LBD_200.csv"
    observations_path = save_path / "Export/additional_observations.json"
    observations = generate_observations(summaries_path, observations_path)

if __name__ == '__main__':
    main()
