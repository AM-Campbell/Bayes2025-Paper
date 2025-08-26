# Export Request

## Preamble

I'm sorry for this large export request. I caught a small error I had made in the code and also added several additional analyses. 

## Files to Export. 

Please export the following files in `56219dua/Project2017/4-Analysis/3-Analysis/AidanPaper/Export/` 

- `test_comparison_0.07.json` (Rerun of Prior Export with Different Parameter)
- `test_comparison_0.05.json` (Rerun of Prior Export)
- `test_comparison_0.03.json` (Rerun of Prior Export with Different Parameter)
- `Table1Info.txt` (NEW)
- `pcdp_hospitals_scatter_plot_data.json` (Rerun of Prior Export)
- `observations_0.07.json` (Rerun of Prior Export with Different Parameter)
- `observations_0.05.json` (Rerun of Prior Export)
- `observations_0.03.json` (Rerun of Prior Export with Different Parameter)
- `most_common_cancer_ccs.csv` (NEW)
- `example_hospital_results_450044_0.05.json` (Rerun of Prior Export with 2 new data elements)
- `example_hospital_results_390111_0.05.json` (Rerun of Prior Export with 2 new data elements)

## Acronym Glossary

- FE(+): Fisher's Exact test positive (p<0.05). Indicates a statistically significant disparity in the overall proportion of patients who received any hospice care at all during the observation window.
- MW(+): Mann-Whitney test positive (p<0.05). Indicates a significant difference in the distribution of the first day of hospice initiation. This test only compares patients who utilized hospice.
- NHD(+): Number of Hospice Days test positive (p<0.05). Indicates a significant difference in the total number of hospice days received. Unlike the MW test, this includes all patients, coding non-users as having 0 days.
- PCD(+): Probable Clinically significant Difference. A Bayesian measure indicating a greater than 95% probability of a meaningful (>5%) disparity in hospice utilization on a specific day.
    - PMPCD(+): A PCD where Patients of Color received More hospice.
    - WMPCD(+): A PCD where White patients received More hospice.
- DBDFE(+): Day-by-Day Fisher's Exact test positive (p<0.05). Indicates a significant disparity in the cumulative proportion of patients who had received hospice by a specific day.

## File Descriptions

### `test_comparison_*.json` 

*The 0.03, 0.05, and 0.07 versions contain the results of the same analysis run with different values of the parameter that determines how large a difference needs to be to be significant.*

*The three files contain varous statistics comparing the findings of the serveral statistical analyses we performed. None describe beneficiary conts or can be used to derive beneficiary counts <= 11.*

- `fep_mwp`: count of hospitals with FE(+) and MW(+)
- `fep_mwn`: count of hospitals with FE(+) and MW(-)
- `fen_mwp`: count of hospitals with FE(-) and MW(+)
- `fen_mwn`: count of hospitals with FE(-) and MW(-)
- `fevmw_festat` and `fevmw_fepval` are the results of a fisher exact test using the above four values in a 2x2 contingency table.
- `pcdp_dbdfep`: count of days with PCD(+) and DBDFE(+)
- `pcdp_dbdfen`: count of days with PCD(+) and DBDFE(-)
- `pcdn_dbdfep`: count of days with PCD(-) and DBDFE(+)
- `pcdn_dbdfen`: count of days with PCD(-) and DBDFE(-)
- `pcdvdbdfe_festat` and `pcdvdbdfe_fepval` are the results of a fisher exact test using the above four values in a 2x2 contingency table.
- `fep_pmpcdp`: count of days with FE(+) and PMPCD(+)
- `fep_wmpcdp`: count of days with FE(+) and WMPCD(+)
- `fep_pcdp`: count of days with FE(+) and PCD(+)
- `fep_pcdn`: count of days with FE(+) and PCD(-)
- `fen_pmpcdp`: count of days with FE(-) and PMPCD(+)
- `fen_wmpcdp`: count of days with FE(-) and WMPCD(+)
- `fen_pcdn`: count of days with FE(-) and PCD(-)
- `fen_pcdp`: count of days with FE(-) and PCD(+)
- `pcdvfe_festat` and `pcdvfe_fepval` are the results of a fisher exact test comparing `fe{p or n}_pcd{p or n}` in a 2x2 contingency table.
- (for `n = 1, 5, 10, 15, 20, 25, 30, 35`)
    - `mwp_n_pcdp`: count of hospitals with MW(+) and $\ge$ n days PCD(+)
    - `mwp_n_pcdn`: count of hospitals with MW(+) and $<$ n days PCD(+)
    - `mwn_n_pcdp`: count of hospitals with MW(-) and $\ge$ n days PCD(+)
    - `mwn_n_pcdn`: count of hospitals with MW(-) and $<$ n days PCD(+)
    - for each of these `mwv{n}pcd_festat` and `mwv{n}pcd_fepval` are the result of applying a fisher exact test to the above four values as a 2x2 contingency table.
- `nhd_mwp_1_pcdp`: count of hospitals with NHD(+) and $\ge$ 1 days PCD(+)
- `nhd_mwp_1_pcdn`: count of hospitals with NHD(+) and $<$ 1 days PCD(+)
- `nhd_mwn_1_pcdp`: count of hospitals with NHD(-) and $\ge$ 1 days PCD(+)
- `nhd_mwn_1_pcdn`: count of hospitals with NHD(-) and $<$ 1 days PCD(+)
- `nhd_mwv1pcd_festat` and `nhd_mwv1pcd_fepval` are the result of applying a fisher exact test to the above four values as a 2x2 contingency table.
- `nhd_mwp_1_wmpcdp`: count of hospitals with NHD(+) and $\ge$ 1 days WMPCD(+)
- `nhd_mwn_1_wmpcdp`: count of hospitals with NHD(-) and $\ge$ 1 days WMPCD(+)
- `nhd_mwp_1_pmpcdp`: count of hospitals with NHD(+) and $\ge$ 1 days PMPCD(+)
- `nhd_mwn_1_pmpcdp`: count of hospitals with NHD(-) and $\ge$ 1 days PMPCD(+)
- `fep_nhd_mwp`: count of hospitals FE(+) and NHD(+)
- `fep_nhd_mwn`: count of hospitals FE(+) and NHD(-)
- `fen_nhd_mwp`: count of hospitals FE(-) and NHD(+)
- `fen_nhd_mwn`: count of hospitals FE(-) and NHD(-)
- `fevnhd_mw_festat` and `fevnhd_mw_fepval` are the result of applying a fisher exact test to the above four values as a 2x2 contingency table.



### `Table1Info.txt` (NEW)

*Reports basic age and sex demographics of the cohort. No ages over 90 or pt. counts under 11 are derivable. This file is quite short and human readable.*

### `pcdp_hospitals_scatter_plot_data.json` 

A list of the subset of hospitals that had at least one PMPCD(+) or WMPCD(+) day along with some associated data for each hospital described below. Each hospital in the analysis had at least 11 white patients and 11 patients of color. The data below describe the number of days for which a disparity was detected, if that disparity favored white patients or patients of color, how many days before death on average the disparities occurred, and if the hospital had a statistically significant difference in the total hospice utilization according to a fisher exact test with p less than 0.05. I believe these data are appropriate to export because they are statistics of hospitals that have greater than 10 white patients and 10 patients of color and none of the data we are requesting to export could be used to derive a count of beneficiaries less than 11. In fact, the data here cant be used to compute any beneficiary count whatsoever.

  - `hospitalID`: The hospital ID
  - `num_pcdp_days`: The number of days identified as WMPCD(+) or PMPCD(+).
  - `avg_days_before_death`: The average number of days before death for the days that were detected as either WMPCD(+) or PMPCD(+)
  - `pcd_type`: Either "WM", white patients received more hospice, or "PM", patients of color received more hospice.
  - `fep`: Boolean value. True if the hospital is "Fisher Exact Positive" meaning a fisher exact test comparing the total utilization rate of hospice by white patients and patients of color yielded a p value less than 0.05.



### `observations_*.json` 


*The 0.03, 0.05, and 0.07 versions contain the results of the same analysis run with different values of the parameter that determines how large a difference needs to be to be significant.*

*These files provide a summary of our results across the cohort of all hospitals which each had at least 11 poc patients and 11 white patients.*

- Num Patients (`num_patients`)
- Num Hospitals (`num_hospitals`)
- Num, % POC, White (`num_poc`, `percent_poc`, `num_white`, `percent_white`)
- `mean_wp`, `median_wp`, `min_wp`, `fq_wp` (first quartile), `tq_wp` (third quartile), `max_wp`
- `mean_pp`, `median_pp`, `min_pp`, `fq_pp`, `tq_pp`, `max_pp`
- Num, % Hospitals with a statistically significant difference in utilization per the Fisher Exact test (p < 0.05) (quality measures approach) (`num_hospitals_fep`, `percent_hospitals_fep`)
- Num, % Hospitals with a positive Mann-Whitney U Test comparing the first day of hospice for those patients who received hospice. (`num_hospitals_mwp`, `percent_hospitals_mwp`)
- Each hospitals had 200 days of patient data and there were `Num Hospitals` hospitals, in this way we count 200\*`Num Hospitals` total days. Of these:
  - Num, % Days PMPCD+ (`num_days_pmpcdp`, `percent_days_pmpcdp`)
  - Num, % Days WMPCD+ (`num_days_wmpcdp`, `percent_days_wmpcdp`)
  - Num, % Days PCD+ (`num_days_pcdp`, `percent_days_pcdp`)
  - Num, % Days DBDFE+ (`num_days_dbdfep`, `percent_days_dbdfep`)
- Num hospitals with at least n = 1, 5, 10, 15, 20, 25, 30, 35 positive days:
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
- Count of hospitals with a positive result for each day since each hospital. Though these series have cell values less than 11, the count is of hospitals which each have at least 11 patients of color and at least 11 white patients. These series can't be used to derive a count of beneifciaries less than 11.
  - `distribution_of_pmpcdp_days` An array of length 200, with each index representing a day, and the value at that index being the number of times that day was PMPCD(+)
  - `distribution_of_wmpcdp_days` The same as above but for WMPCD(+)
  - `distribution_of_dbdfep_days` The same as above but from DBDFE(+)


### `most_common_cancer_ccs.csv` (NEW)

A table of the prevalence of each CCS cancer diagnosis category in our patient population. Each count of patients, and it's complement in the total (total - count) includes greater than 11 patients. Using the data in this table along with all previously exported data as well as all data in this export request, it is not possible to derive a pt. count of 11 or fewer. The counts in each row are not mutually exclusive so each row can be evaluated againt export critera independently. The rows can not be used together to dervie pt. counts less than or equal to 11. 


### `example_hospital_results_*_0.05.json` (2 NEW elements)

*Includes results from a single hospital and provides the following variables.*

  - `hospitalID`: int. the hospital id of the example hospital
  - `n_w`: int. number of white patients at the hospital (much greater than 11)
  - `n_p`: int. number of patients of color at the hospital (much greater than 11)
  - `u_w`: int. number of white patients at the hospital who utilized hospice (much greater than 11)
  - `u_p`: int. number of patients of color at the hospital who utilized hospice (much greater than 11)
  - `fep` float. fisher exact p value. (a statistic of the entire hospital which has more than 11 patients)
  - `fes` float. fisher exact statistic. (a statistic of the entire hospital which has more than 11 patients) 
  - `mwp` float. Mann Whitney P value. (a statistic of the entire hospital which has more than 11 patients)
  - `mws` float. Mann Whitney Statistic (a statistic of the entire hospital which has more than 11 patients)
  - `day_ind_on_0_199` list. The indices of days exported. A single continuous slice of [0,199]. Has the same length as each list in this json file. 
  - `min_beneficiary_count` list. The minimum of (the number of white patients who received hospice by day $j$, the number of poc patients who recieved hospice by day $j$) for each exported day $j$. This is provided as a check. You can see that each value is greater than 11. This implies that the non-exported days (i.e. those days not included in `day_ind_on_0_199`) had fewer than 11 patients but the specific count is not derivable from any exported data.
  - `min_non_beneficiary_count` list. The minimum, for each day, of white or poc patients who did not receive hospice for each exported day. We took care that both the number of beneficiaries and non-beneficiaries in each group white and poc were more than 11. 
  - `diffSignal` list. The difference signal for the exported days. The fraction of white patients who recieved hospice by day $j$ minus the fraction of poc patients who recieved hospice by day $j$. As ensured by the above two series this doesn't describe or allow for the derviation of a patient count less than 11.
  - `cilow` list. The lower credible interval series for the exported days. The lower 95% credible interval of the difference singal. 
  - `cihigh` list. The higher credible interval series for the exported days. The upper 95% credible interval of the difference singal.
  - `dbdfe` list. The series of dbdfe p values for the exported days. A fisher exact test to compare the fraction of white patients who had hospice by day $j$ with the fraction of poc patients who did so. Again only for the exported days which each have a >11 patients in each of the four categories hospice&poc, hospice&white, nohospice&poc, nohospice&white.
  - `dbdfe_stat` list. The series of day by day fisher exact statistics for the exported days. The fisher exact statistic produced by the above test.
  - `wmpcd` list. The series of probabilities that the difference signal was greater than 0.05 for each exported day. A statistic of the counts for each of the four categories hospice&poc, hospice&white, nohospice&poc, nohospice&white for each day which each have >11 patients as ensured by `min_beneficiary_count` and `min_non_beneficiary_count`
  - `pmpcd` list. The series of probabilities that the difference signal was less than -0.05 for each exported day. Safe to export for the same reason that `wmpcd` is.
  - `w_util` list. The fraction of white patients who utilized hospice for each exported day. Safe to export becuase for each day `min_beneficiary_count` and `min_non_beneficiary_count` are >11.
  - `p_util` list. The fraction of poc patients who utilized hospice for each exported day. Safe to export because for each day `min_beneficiary_count` and `min_non_beneficiary_count` are >11.
  - (NEW) `nhd_mwp` float. Mann Whitney P value for the Number of Hospice Days Comparison. (a statistic of the entire hospital which has more than 11 patients)
  - (NEW) `nhd_mws` float. Mann Whitney Statistic for the Number of Hospice Days Comparison. (a statistic of the entire hospital which has more than 11 patients)

