# Paper

## Results

### Example Results Figure

- `Results Figure` showing all the results for a given hospital (get by hospitalID)
- The results figure will be populated with the data from `example_hospital_results.json` which is specified in the appendix section of this document.

### Observations

> Percents are always given as percents $[0, 100]$, not fractions $[0,1]$

- Num Patients (`num_patients`)
- Num Hospitals (`num_hospitals`)
- Num, % POC, White (`num_poc`, `percent_poc`, `num_white`, `percent_white`)
- `mean_wp`, `median_wp`, `min_wp`, `fq_wp`, `tq_wp`, `max_wp`
- `mean_pp`, `median_pp`, `min_pp`, `fq_pp`, `tq_pp`, `max_pp`
- Num, % Hospitals with a statistically significant difference in utilization per the Fisher Exact test (p < 0.05) (quality measures approach) (`num_hospitals_fep`, `percent_hospitals_fep`)
- Num, % Hospitals with a positive Mann-Whitney U Test comparing the first day of hospice for those patients who received hospice. (`num_hospitals_mwp`, `percent_hospitals_mwp`)
- Each hospitals had 200 days of patient data and there were `Num Hospitals` hospitals, in this way we count 200\*`Num Hospitals` total days. Of these:
  - Num, % Days PMPCD+ (`num_days_pmpcdp`, `percent_days_pmpcdp`)
  - Num, % Days WMPCD+ (`num_days_wmpcdp`, `percent_days_wmpcdp`)
  - Num, % Days PCD+ (`num_days_pcdp`, `percent_days_pcdp`)
  - Num, % Days DBDFE+ (`num_days_dbdfep`, `percent_days_dbdfep`)
- Num hospitals with at least n = 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 positive days:
  - PCD
    - `num_hospitals_with_{n}_pcdp_day`
    - `percent_hospitals_with_{n}_pcdp_day`
  - PMPCD
    - `num_hospitals_with_{n}_pmpcdp_day`
    - `percent_hospitals_with_{n}_pmpcdp_day`
  - WMPCD
    - `num_hospitals_with_{n}_wmpcdp_day`
    - `percent_hospitals_with_{n}_wmpcdp_day`
  - DBDFE
    - `num_hospitals_with_{n}_dbdfep_day`
    - `percent_hospitals_with_{n}_dbdfep_day`
- Did any hospital have a day that was PMPCD+ and WMPCD+? (`is_hospital_with_pmpcdp_and_wmpcdp`). Either True or False.
- Stacked histogram of PMPCD+ and WMPCD+ days to show the days that were most positive. Show the same histogram on the same chart for DBDFE+ days but as a scatter plot.
  - `distribution_of_pmpcdp_days` An array of length 200, with each index representing a day, and the value at that index being the number of times that day was PMPCD(+)
  - `distribution_of_wmpcdp_days` The same as above but for WMPCD(+)
  - `distribution_of_dbdfep_days` The same as above but from DBDFE(+)


### Test Comparison

1. FE vs MW as a hospital classifier.

   - Contingency Table w/ FE Test
   - **Key Observation:** The two tests don't agree.
   - **Purpose:** You are missing something by neglecting the temporal component.
   - Variables:
     - `fep_mwp`: count of hospitals with FE(+) and MW(+)
     - `fep_mwn`: count of hospitals with FE(+) and MW(-)
     - `fen_mwp`: count of hospitals with FE(-) and MW(+)
     - `fen_mwn`: count of hospitals with FE(-) and MW(-)

2. PCD(+/-) vs DBDFE(+/-) as a day classifier.

   - Contingency Table w/ FE Test
   - **Key Observation:** These largely agree, though DBDFE is more sensitive (maybe less specific) expect for small sample sizes where DBDFE may fail to reject the null but the PCD(+)... 
   - **Purpose:**: We conclude that if you just want to know yay or nay, these two tests can be used pretty much interchangably, though if you want to know a meaningful probability, you should go with the bayesian approach (also, I'm somewhat biased to the bayesian method because it's more reasonable). 
   - Variables:
     - `pcdp_dbdfep`: count of days with PCD(+) and DBDFE(+)
     - `pcdp_dbdfen`: count of days with PCD(+) and DBDFE(-)
     - `pcdn_dbdfep`: count of days with PCD(-) and DBDFE(+)
     - `pcdn_dbdfen`: count of days with PCD(-) and DBDFE(-)

3. FE vs PCD+ (PMPCD+, WMPCD+) as a day classifier.

   - Nested donut chart
   - Contingency Table w/ FE Test (comparing FE & PCD+)
   - **Key Observation:** The two tests don't agree.
   - **Purpose:** Again, you are missing something by neglecting the temporal component. Adds weight to the purpose of the comparison FE vs MW.
   - Variables:
     - `fep_pmpcdp`: count of days with FE(+) and PMPCD(+)
     - `fep_wmpcdp`: count of days with FE(+) and WMPCD(+)
     - `fep_pcdp`: count of days with FE(+) and PCD(+)
     - `fep_pcdn`: count of days with FE(+) and PCD(-)
     - `fen_pmpcdp`: count of days with FE(-) and PMPCD(+)
     - `fen_wmpcdp`: count of days with FE(-) and WMPCD(+)
     - `fen_pcdn`: count of days with FE(-) and PCD(-)
     - `fen_pcdp`: count of days with FE(-) and PCD(+)

4. MW vs $\ge$ 1, 5, and 10 days PCD as a hospital classifier.
   - Contingency Table w/ FE Test
   - **Key Observation:**: I don't know the result of this (though I have computed it). We would hope they agree.
   - **Purpose:** Answer the question, can we just use MW to screen for inequities in the temporal dimension (it would be easier if true and we would say in the discussion that people can use MW first).
   - Variables: (for `n = 1, 5, 10`)
   - `mwp_n_pcdp`: count of hospitals with MW(+) and $\ge$ n days PCD(+)
   - `mwp_n_pcdn`: count of hospitals with MW(+) and $<$ n days PCD(+)
   - `mwn_n_pcdp`: count of hospitals with MW(-) and $\ge$ n days PCD(+)
   - `mwn_n_pcdn`: count of hospitals with MW(-) and $<$ n days PCD(+)

## Methods

### Dataset / Cohort

- Describe the dataset and cohort.

### Series, Statistics, and Tests

- `Methods Figure`
- Description of everything on the methods figure in detail. (Should follow the same structure.)

### Computation of Observations and Test Comparison

#### On the DAC

`methods.py` controls the top level execution of all paper code _on the DAC_. It does the following. Each file can be run manually in order with the same effect as running `methods.py`.

1. Runs `generate_data(save_path, function)` in `get_summaries.py` to generate `summaries.csv`.
2. Runs code to generate JSON file with observations. No plotting is performed.
   1. Runs `generate_observations` via `main()` in `all_observations.py`. This produces `observations.json` which includes all the variables in the observations section above.
   2. Runs `generate_test_comparison_observations` via `main()` in `test_comparison.py`. This produces `test_comparison.json`.
   3. Runs `main()` in `get_example_hospital_results.py`. This produces `example_hospital_results.json`.

`test_comparison.json`, `obervations.json`, and `example_hospital_results.json` can be exported from the DAC.

#### Off the DAC

... TODO

### Comparing The Tests

- Describe the computation of the statistics used for each of the comparisons above.

## Appendix

### Format of `summaries.csv`

Each row represents a hospital.

**Columns**:

- '': int. index. Can be ignored.
- `hospitalID`: int.
- `function`: string. This will always be 'HospiceTreat' for this paper.
- `n_w`: int. number of white patients at the hospital
- `n_p`: int. number of patients of color at the hospital
- `u_w`: int. number of white patients at the hospital who utilized hospice
- `u_p`: int. number of patients of color at the hospital who utilized hospice
- `count_dbdfe_sig`: int. number of days DBDFE(+)
- `dbdfe_sig_days`: bracketed, space separated list of int. Days [0-199] where DBDFE(+)
- `count_wmpcd_sig`: int. number of days WMPCD(+)
- `wmpcd_sig_days`: bracketed, space separated list of int. Days [0-199] where WMPCD(+)
- `count_pmpcd_sig`: int. number of days PMPCD(+)
- `pmpcd_sig_days`: bracketed, space separated list of int. Days [0-199] where PMPCD(+)
- `mwp`: float. Mann-Whitney U Test p-value
- `qmp`: float. Fisher Exact p-value
- `fep_bp`: int. days DBDFE(+), PCD(+)
- `fep_bn`: int. days DBDFE(+), PCD(-)
- `fen_bp`: int. days DBDFE(-), PCD(+)
- `fen_bn`: int. days DBDFE(-), PCD(-)

### Format of `observations.json`

`observations.json` is a flat dictionary where the keys include only and all the variable names listed in the observations section above.

### Format of `example_hospital_results.json`

- `n_w`: int. number of white patients at the hospital
- `n_p`: int. number of patients of color at the hospital
- `u_w`: int. number of white patients at the hospital who utilized hospice
- `u_p`: int. number of patients of color at the hospital who utilized hospice
- `fep` float. fisher exact p value.
- `fes` float. fisher exact statistic.
- `mwp` float. Mann Whitney P value.
- `mws` float. Mann Whitney Statistic
- `day_ind_on_0_199` list. The indices of days exported. A single continuous slice of [0,199]. Has the same length as each list in this json file.
- `min_beneficiary_count` list. The minimum of the number of white or poc patients who received hospice for each exported day.
- `min_non_beneficiary_count` list. The minimum, for each day, of white or poc patients who did not receive hospice for each exported day.
- `diffSignal` list. The difference signal for the exported days.
- `cilow` list. The lower credible interval series for the exported days.
- `cihigh` list. The higher credible interval series for the exported days
- `dbdfe` list. The series of dbdfe p values for the exported days.
- `dbdfe_stat` list. The series of day by day fisher exact statistics for the exported days (not used).
- `wmpcd` list. The series of probabilities that the difference signal was greater than 0.05 for each exported day.
- `pmpcd` list. The series of probabilities that the difference signal was less than -0.05 for each exported day.
- `w_util` list. The fraction of white patients who utilized hospice for each exported day.
- `p_util` list. The fraction of poc patients who utilized hospice for each exported day.

