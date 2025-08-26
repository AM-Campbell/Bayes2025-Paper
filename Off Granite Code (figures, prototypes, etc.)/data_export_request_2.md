# Data Export Request

Requestor: Aidan Campbell
User Type: Dartmouth Student
Project Number: 56219
Transfer Type: Transfer Out

Description: Transfer out

## Description of data files

- `example_hospital_results_*.json`: Includes results from a single hospital and provides the following variables. (* should be an integer and there should be 2 such files to export)
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

- `hospitalIDs.txt` - A text file with the hospital IDs included in the analysis. It will be used to do an analysis of hospital properties off the DAC (e.g. what percent are academic centers, rural hospitals)

- `pcdp_hospitals_scatter_plot_data.json` is a list of the subset of hospitals that had at least one PMPCD(+) or WMPCD(+) day along with some associated data for each hospital described below. Each hospital in the analysis had at least 11 white patients and 11 patients of color. The data below describe the number of days for which a disparity was detected, if that disparity favored white patients or patients of color, how many days before death on average the disparities occurred, and if the hospital had a statistically significant difference in the total hospice utilization according to a fisher exact test with p less than 0.05. I believe these data are appropriate to export because they are statistics of hospitals that have greater than 11 white patients and 11 patients of color and none of the data we are requesting to export could be used to derive a count of beneficiaries less than 11. In fact, the data here cant be used to compute any beneficiary count whatsoever.
  - `hospitalID`: The hospital ID
  - `num_pcdp_days`: The number of days identified as WMPCD(+) or PMPCD(+).
  - `avg_days_before_death`: The average number of days before death for the days that were detected as either WMPCD(+) or PMPCD(+)
  - `pcd_type`: Either "WM", white patients received more hospice, or "PM", patients of color received more hospice.
  - `fep`: Boolean value. True if the hospital is "Fisher Exact Positive" meaning a fisher exact test comparing the total utilization rate of hospice by white patients and patients of color yielded a p value less than 0.05.

## Clarification of Terminology

- FE (+) and (-): A hospital is FE(+) if a fisher exact test comparing the fraction of white patients who received hospice to the fraction of patients of color who recieved hospice is positive (p < 0.05) and FE(-) otherwise.
- MW (+) and (-): A hospital is MW(+) if a Mann-Whitney U test comparing the distribution of the first day of hospice for those patients who received hospice for white patients and patients of color is significant (p < 0.05). MW(-) otherwise.
- PMPCD(+) and (-): A day is PMPCD(+) if the probability that the difference difference signal is less than -0.05 is greater than 95% and PMPCD(-) otherwise.
- WMPCD(+) and (-): A day is WMPCD(+) if the probability that the difference signal is greater than 0.05 is greater than 95% and WMPCD(-) otherwise.
- PCD(+) and (-): A day is PCD(+) if it is either PMPCD(+) or WMPCD(+) and PCD(-) otherwise.
