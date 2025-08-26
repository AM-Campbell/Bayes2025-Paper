import get_summaries 
import test_comparison 
import all_observations 
import get_example_hospital_results
from util import generate_n_file
import get_table_1_results
import get_hospital_scatter_plot_data
import additional_observations

TEST = False 

PCD_THRESH=0.05
LOOKBACK_DAYS=100
generate_n_file('HospiceTreat', LOOKBACK_DAYS)
get_summaries.main(PCD_THRESH=PCD_THRESH, LOOKBACK_DAYS=LOOKBACK_DAYS, TEST=TEST)
all_observations.main(PCD_THRESH=PCD_THRESH, LOOKBACK_DAYS=LOOKBACK_DAYS)
test_comparison.main(PCD_THRESH=PCD_THRESH, LOOKBACK_DAYS=LOOKBACK_DAYS)


PCD_THRESH=0.05
LOOKBACK_DAYS=200
generate_n_file('HospiceTreat', LOOKBACK_DAYS)
get_summaries.main(PCD_THRESH=PCD_THRESH, LOOKBACK_DAYS=LOOKBACK_DAYS, TEST=TEST)
all_observations.main(PCD_THRESH=PCD_THRESH, LOOKBACK_DAYS=LOOKBACK_DAYS)
test_comparison.main(PCD_THRESH=PCD_THRESH, LOOKBACK_DAYS=LOOKBACK_DAYS)


PCD_THRESH=0.03
LOOKBACK_DAYS=200
generate_n_file('HospiceTreat', LOOKBACK_DAYS)
get_summaries.main(PCD_THRESH=PCD_THRESH, LOOKBACK_DAYS=LOOKBACK_DAYS, TEST=TEST)
all_observations.main(PCD_THRESH=PCD_THRESH, LOOKBACK_DAYS=LOOKBACK_DAYS)
test_comparison.main(PCD_THRESH=PCD_THRESH, LOOKBACK_DAYS=LOOKBACK_DAYS)


PCD_THRESH=0.07
LOOKBACK_DAYS=200
generate_n_file('HospiceTreat', LOOKBACK_DAYS)
get_summaries.main(PCD_THRESH=PCD_THRESH, LOOKBACK_DAYS=LOOKBACK_DAYS, TEST=TEST)
all_observations.main(PCD_THRESH=PCD_THRESH, LOOKBACK_DAYS=LOOKBACK_DAYS)
test_comparison.main(PCD_THRESH=PCD_THRESH, LOOKBACK_DAYS=LOOKBACK_DAYS)


# Other
get_hospital_scatter_plot_data.main()
additional_observations.main()
get_table_1_results.main()
