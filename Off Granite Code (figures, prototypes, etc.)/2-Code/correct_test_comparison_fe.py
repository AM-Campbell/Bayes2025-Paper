# This file corrects the test_comparison.json file in 1-Input/Figure-and-Paper-Data
# specifically it changes pcdvdbdfe_fepval to fix it from the previous incorrect calculation
# that was performed on the dac and exported. 

# The incorrect code is as follows:
# The following line shows the incorrect code for reference and should not be compiled:
# comparisons['pcdvdbdfe_festat'], comparisons['pcdvdbdfe_fepval'] = stats.fisher_exact(
#         np.array(
#             [
#                 [pcdp_dbdfep, pcdp_dbdfen],
#                 [pcdn_dbdfep, pcdp_dbdfen]
#              ]
#             )
#         )

# we now read in test_comparison.json and correct the pcdvdbdfe_fepval value

import json
import numpy as np
from scipy import stats

with open("1-Input/Figure-and-Paper-Data/test_comparison.json", "r") as f:
    comparisons = json.load(f)

pcdp_dbdfep = comparisons["pcdp_dbdfep"]
pcdp_dbdfen = comparisons["pcdp_dbdfen"]
pcdn_dbdfep = comparisons["pcdn_dbdfep"]
pcdn_dbdfen = comparisons["pcdn_dbdfen"]

contingency = np.array(
    [
        [pcdp_dbdfep, pcdp_dbdfen],
        [pcdn_dbdfep, pcdn_dbdfen]
    ]
)

pcdvdbdfe_festat, pcdvdbdfe_fepval = stats.fisher_exact(contingency)


print(contingency)
print(f'{pcdvdbdfe_fepval:.5e}')
print(f'{pcdvdbdfe_festat:.5e}')

comparisons['pcdvdbdfe_festat'], comparisons['pcdvdbdfe_fepval'] = pcdvdbdfe_festat, pcdvdbdfe_fepval

with open("1-Input/Figure-and-Paper-Data/test_comparison.json", "w") as f:
    json.dump(comparisons, f, cls=json.JSONEncoder)