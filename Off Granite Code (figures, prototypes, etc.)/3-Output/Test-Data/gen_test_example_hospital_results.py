# frequency = 0.02  # Adjust frequency for lower/higher oscillations

# data = {
#     "n_w": 150,
#     "n_p": 100,
#     "u_w": 30,
#     "u_p": 20,
#     "fep": 0.045,
#     "fes": 1.75,
#     "mwp": 0.035,
#     "mws": 2.1,
#     "day_ind_on_0_199": list(range(start_day, end_day + 1)),
#     "min_beneficiary_count": np.round(np.linspace(5, 10, num_days), 2).tolist(),  # Linear
#     "min_non_beneficiary_count": np.round(np.linspace(3, 7, num_days), 2).tolist(),  # Linear
#     "diffSignal": np.round(0.1 * np.sin(frequency * np.arange(num_days)), 2).tolist(),  # Sinusoidal
#     "cilow": np.round(-0.1 + 0.05 * np.sin(frequency * np.arange(num_days)), 2).tolist(),  # Sinusoidal
#     "cihigh": np.round(0.05 * np.sin(frequency * np.arange(num_days)), 2).tolist(),  # Sinusoidal
#     "dbdfe": np.round(0.03 + 0.01 * np.sin(frequency * np.arange(num_days)), 2).tolist(),  # Sinusoidal
#     "dbdfe_stat": np.round(1.25 + 0.2 * np.sin(frequency * np.arange(num_days)), 2).tolist(),  # Sinusoidal
#     "wmpcd": np.round(0.9 + 0.02 * np.sin(frequency * np.arange(num_days)), 2).tolist(),  # Sinusoidal
#     "pmpcd": np.round(0.1 + 0.02 * np.sin(frequency * np.arange(num_days)), 2).tolist(),  # Sinusoidal
#     "w_util": np.round(0.2 + 0.05 * np.sin(frequency * np.arange(num_days)), 2).tolist(),  # Sinusoidal
#     "p_util": np.round(0.15 + 0.05 * np.sin(frequency * np.arange(num_days)), 2).tolist(),  # Sinusoidal
# }

import numpy as np
import json

# Function for calculating rolling average
def rolling_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Set parameters
start_day = 130
end_day = 199
num_days = end_day - start_day + 1

# Generate fixed values
ds = np.round(np.random.uniform(-1.05, 0.1, num_days), 2)
cil = ds - 0.3
cih = ds + 0.3

# Generate original data
data = {
    "hospitalID": 1384329,
    "n_w": 150,
    "n_p": 100,
    "u_w": 30,
    "u_p": 20,
    "fep": 0.045,
    "fes": 1.75,
    "mwp": 0.035,
    "mws": 2.1,
    "day_ind_on_0_199": list(range(start_day, end_day + 1)),
    "min_beneficiary_count": np.random.randint(5, 10, num_days).tolist(),
    "min_non_beneficiary_count": np.random.randint(3, 7, num_days).tolist(),
    "diffSignal": ds.tolist(),
    "cilow": cil.tolist(),
    "cihigh": cih.tolist(),
    "dbdfe": np.round(np.random.uniform(0.01, 0.1, num_days), 2).tolist(),
    "dbdfe_stat": np.round(np.random.uniform(1, 1.5, num_days), 2).tolist(),
    "wmpcd": np.round(np.random.uniform(0.5, 1, num_days), 2).tolist(),
    "pmpcd": np.round(np.random.uniform(0.00, 0.5, num_days), 2).tolist(),
    "w_util": np.round(np.random.uniform(0.15, 0.25, num_days), 2).tolist(),
    "p_util": np.round(np.random.uniform(0.1, 0.2, num_days), 2).tolist()
}

# Apply smoothing to specific fields
window_size = 5  # Set the rolling window size

# Smooth each specified field
data['dbdfe'] = rolling_average(data['dbdfe'], window_size).tolist()
data['wmpcd'] = rolling_average(data['wmpcd'], window_size).tolist()
data['pmpcd'] = rolling_average(data['pmpcd'], window_size).tolist()
data['w_util'] = rolling_average(data['w_util'], window_size).tolist()
data['p_util'] = rolling_average(data['p_util'], window_size).tolist()

# Adjust the length of day_ind_on_0_199 and other lists if necessary to match the smoothed data length
smoothed_num_days = len(data['dbdfe'])  # New length after smoothing
data["day_ind_on_0_199"] = data["day_ind_on_0_199"][:smoothed_num_days]
data["min_beneficiary_count"] = data["min_beneficiary_count"][:smoothed_num_days]
data["min_non_beneficiary_count"] = data["min_non_beneficiary_count"][:smoothed_num_days]
data["diffSignal"] = data["diffSignal"][:smoothed_num_days]
data["cilow"] = data["cilow"][:smoothed_num_days]
data["cihigh"] = data["cihigh"][:smoothed_num_days]
data["dbdfe_stat"] = data["dbdfe_stat"][:smoothed_num_days]

# The `data` dictionary now contains smoothed values for the specified fields


# Save as JSON
json_output = json.dumps(data, indent=2)
with open("3-Output/Test-Data/example_hospital_results.json", "w") as file:
    file.write(json_output)
