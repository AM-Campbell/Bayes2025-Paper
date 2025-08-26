from typing import Literal, TypedDict
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class UtilizationData(TypedDict):
    n_a: int
    n_b: int
    data_range: tuple[int, int]
    num_bins: int
    data_a: list[int | float]
    data_b: list[int | float]
    a_name: str
    b_name: str
    x_axis_label: str
    series_name: str

class UtilizationDataGenerator():
    
    distribution_a: np.ndarray
    distribuiton_b: np.ndarray
    data_range: tuple[int, int]
    num_bins: int
    a_name: str
    b_name: str
    x_axis_label: str
    series_name: str

    current_n_a: int | None = None
    current_n_b: int | None = None
    current_data_a: list[int | float] | None = None
    current_data_b: list[int | float] | None = None
    current_utilization_fraction_a: float | None = None
    current_utilization_fraction_b: float | None = None
    
    def __init__(self, 
                 distribution_a: np.ndarray, 
                 distribution_b: np.ndarray, 
                 data_range: tuple[int, int], 
                 num_bins: int, 
                 a_name: str, 
                 b_name: str,
                 x_axis_label: str, 
                 series_name: str
                 ):

        self.distribution_a = distribution_a
        self.distribution_b = distribution_b
        self.data_range = data_range
        self.num_bins = num_bins
        self.a_name = a_name
        self.b_name = b_name
        self.x_axis_label = x_axis_label
        self.series_name = series_name
    
    def generate_data(self, n_a: int, n_b: int, utilization_fraction_a: float, utilization_fraction_b: float) -> UtilizationData:
        self.current_utilization_fraction_a = utilization_fraction_a
        self.current_utilization_fraction_b = utilization_fraction_b
        self.current_n_a = n_a
        self.current_n_b = n_b

        self.current_data_a = np.random.choice(np.linspace(self.data_range[0], self.data_range[1]-1, len(self.distribution_a)), 
                                               int(self.current_n_a*self.current_utilization_fraction_a), p=self.distribution_a)
        
        self.current_data_b = np.random.choice(np.linspace(self.data_range[0], self.data_range[1]-1, len(self.distribution_b)),
                                                  int(self.current_n_b*self.current_utilization_fraction_b), p=self.distribution_b)

        return {
            'n_a': self.current_n_a,
            'n_b': self.current_n_b,
            'data_range': self.data_range,
            'num_bins': self.num_bins,
            'data_a': self.current_data_a,
            'data_b': self.current_data_b,
            'a_name': self.a_name,
            'b_name': self.b_name,
            'x_axis_label': self.x_axis_label,
            'series_name': self.series_name
        }
        
        
    def get_distributions(self):
        return self.distribution_a, self.distribution_b
        
  
from typing import Callable, Optional, Tuple
def create_utilization_signals(data: UtilizationData, smoothing_fn: Optional[Callable] = None) -> tuple[np.ndarray, np.ndarray]:
    """Creates matrices of utilization signals for each group. Each row in the matrix is the utilization signal of a single entry in the group. 
        The utilization signal is a binary signal that is 1 if the utilization is less than or equal to the bin value and 0 otherwise.


    Args:
        data (UtilizationData): A dictionary containing the following keys:
            n_a (int): Number of entries in group A.
            n_b (int): Number of entries in group B.
            data_range (tuple[int|float, int|float]): A tuple containing the minimum (inclusive) and maximum (exclusive) values of the data. i.e. [min, max), to remain consistent with python ranges.
            num_bins (int): Number of partitions of the data range.
            data_a (list[int | float]): A list of utilization days for group A.
            data_b (list[int | float]): A list of utilization days for group B.
            a_name (str): Name of group A.
            b_name (str): Name of group B.
            series_name (str): Name of the series.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The first array contains the utilization signals for group A and is n_a x num_bins and the second array contains the utilization signals for group B and is n_b x num_bins. For each x in data_group, the value of the signal at x is 1 if x is in the bin, 0 before and 1 thereafter.
    """

    # Create bins
    bins = np.linspace(data['data_range'][0], data['data_range'][1]-1, data['num_bins']) # -1 to avoid including the maximum value in the last bin since the last bin is a closed interval
        
    # Create signals
    signals_a = np.zeros((data['n_a'], data['num_bins']))
    for i, x in enumerate(data['data_a']):
        signals_a[i] = (x <= bins).astype(int)
    signals_b = np.zeros((data['n_b'], data['num_bins']))
    for i, x in enumerate(data['data_b']):
        signals_b[i] = (x <= bins).astype(int)

    if smoothing_fn is not None:
        signals_a = smoothing_fn(signals_a)
        signals_b = smoothing_fn(signals_b)

    return signals_a, signals_b


def create_utilization_signals_sanity_check():
    data = {
        'n_a': 7,
        'n_b': 3,
        'data_range': (0, 5),
        'num_bins': 5,
        'data_a': [],
        'data_b': [2, 4],
        'a_name': 'Group A',
        'b_name': 'Group B',
        'series_name': 'Utilization'
    }

    s_a, s_b = create_utilization_signals(data)
    print(s_a)
    print(s_b)
#create_utilization_signals_sanity_check() 

def calculate_difference_signal(signals_a: np.ndarray, signals_b: np.ndarray) -> np.ndarray:
    """Calculates the difference signal between two groups. The difference signal is the difference between the utilization signals of the two groups which is the mean of the utilization signals of group A minus the mean of the utilization signals of group B.

    Args:
        tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The first array contains the utilization signals for group A and the second array contains the utilization signals for group B. 

    Returns:
        np.ndarray: A numpy array containing the difference signal between the two groups. The difference signal is the difference between the utilization signals of the two groups.
    """
    return np.mean(signals_a, axis=0) - np.mean(signals_b, axis=0)

def calculate_difference_signal_sanity_check():
    signals_a = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
    signals_b = np.array([[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]])
    diff = calculate_difference_signal(signals_a, signals_b)
    print(diff)
#calculate_difference_signal_sanity_check()

def boostrap_difference_signal_sampling_distribution(signals_a: np.ndarray, 
                                                     signals_b: np.ndarray, 
                                                     num_resamples: int = 1000) -> np.ndarray:
    """Approximates the sampling distribution of the difference signal between two groups using the bootstrap method. 
        The sampling distribution is the distribution of the difference between the utilization signals of the two groups.

    Args:
        signals_a (np.ndarray): A numpy array containing the utilization signals for group A.
        signals_b (np.ndarray): A numpy array containing the utilization signals for group B.
        num_resamples (int, optional): Number of resamples to generate. Defaults to 1000.

    Returns:
        np.ndarray: A numpy array containing the sampling distribution of the difference signal between the two groups.
    """
    assert signals_a.shape[1] == signals_b.shape[1], "The number of bins in the utilization signals of the two groups must be the same."
    
    len_a = signals_a.shape[0]
    len_b = signals_b.shape[0]
    
    boostrapped_difference_signal_sampling_distribution = np.empty((num_resamples, signals_a.shape[1]))
    for sample_idx in range(num_resamples):
        resampled_signals_a = signals_a[np.random.choice(len_a, len_a, replace=True)]
        resampled_signals_b = signals_b[np.random.choice(len_b, len_b, replace=True)]
        boostrapped_difference_signal_sampling_distribution[sample_idx] = calculate_difference_signal(resampled_signals_a, resampled_signals_b)

    return boostrapped_difference_signal_sampling_distribution


def get_confidence_intervals_from_sampling_distribution(sampling_distribution: np.ndarray, 
                                                        confidence_level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the confidence intervals from the sampling distribution of the difference signal between two groups.

    Args:
        sampling_distribution (np.ndarray): A numpy array containing the sampling distribution of the difference signal between two groups.
        confidence_level (float, optional): Confidence level. Defaults to 0.95.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the lower and upper bounds of the confidence interval for 
                                        each index in range sampling_distribution.shape[1].
    """

    ci_lower, ci_upper = np.quantile(sampling_distribution, [(1-confidence_level)/2, confidence_level + (1-confidence_level)/2], axis=0)
    return ci_lower, ci_upper

def get_p_difference_signal_greater_than_threshold_in_each_direction(sampling_distribution: np.ndarray, 
                                                                     threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the probability of the difference signal being greater than the threshold in each direction.

    Args:
        sampling_distribution (np.ndarray): A numpy array containing the sampling distribution of the difference signal between two groups.
        threshold (float): The threshold value on the interval [min(signals_a_{i,j}) - max(signals_b_{i,j}), max(signals_a_{i,j}) - min(signals_b_{i,j})] for each j and all i.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the probability of the difference signal being greater than the threshold in each direction.
    """
    p_greater_than_threshold_favoring_a = np.mean(sampling_distribution > threshold, axis=0)
    p_greather_than_threshold_favoring_b = np.mean(sampling_distribution < -threshold, axis=0)
    return p_greater_than_threshold_favoring_a, p_greather_than_threshold_favoring_b

def count_beneficiaries(signals: np.ndarray):
    """Checks for days with < 11 and > 0 beneficiaries.

    Args:
        signals (np.ndarray): A numpy array containing the utilization signals for a group.
    """

    # count the non-zero entries for each column (day)
    num_beneficiaries = np.count_nonzero(signals, axis=0)
    # requires_supression = np.logical_and(num_beneficiaries < 11, num_beneficiaries > 0)
    # if np.any(requires_supression):
        # print(f"Days with < 11 and > 0 beneficiaries: {np.where(requires_supression)[0]}")

    return num_beneficiaries
    

def analyze_utilization_data(data: UtilizationData, 
                             distribution_a: np.ndarray | None = None, 
                             distribution_b: np.ndarray | None = None, 
                             utilization_fraction_a: float | None = None, 
                             utilization_fraction_b: float | None = None,
                             smoothing_fn: Optional[Callable] = None,
                             smoothing_kernel_message: Optional[str] = None
                             ):

    CI_P_VALUE = 0.95
    DIFFERENCE_THRESHOLD = 0.05
    
    signals_a, signals_b = create_utilization_signals(data, smoothing_fn)

    # num_beneficiaries_a = count_beneficiaries(signals_a)
    # num_beneficiaries_b = count_beneficiaries(signals_b)
    # difference_signal_beneficiaries = num_beneficiaries_a + num_beneficiaries_b
    # difference_require_supression = np.logical_and(difference_signal_beneficiaries < 11, difference_signal_beneficiaries > 0)
    # print(f"Days with < 11 and > 0 beneficiaries in difference signal: {np.where(difference_require_supression)[0]}")

    difference_signal = calculate_difference_signal(signals_a, signals_b)
    approximate_sampling_distribution = boostrap_difference_signal_sampling_distribution(signals_a, signals_b)
    ci_lower, ci_upper = get_confidence_intervals_from_sampling_distribution(approximate_sampling_distribution, CI_P_VALUE)
    p_greater_than_threshold_favoring_a, p_greather_than_threshold_favoring_b = get_p_difference_signal_greater_than_threshold_in_each_direction(approximate_sampling_distribution, DIFFERENCE_THRESHOLD)

    # Plotting the Results
    fig, axs = plt.subplots(2, 1, figsize=(10,12))
    fig.suptitle(f'{data["series_name"]} Analysis {smoothing_kernel_message}')

    axs[0].set_title('Difference Signal')

    axs[0].plot(difference_signal, label=f'Difference Signal ({data["a_name"]} - {data["b_name"]})')
    axs[0].fill_between(range(len(difference_signal)), ci_lower, ci_upper, alpha=0.3, label=f'{CI_P_VALUE} Confidence Interval')
    axs[0].axhline(y=DIFFERENCE_THRESHOLD, color='r', linestyle='--', label=f'Threshold: {DIFFERENCE_THRESHOLD}')
    axs[0].axhline(y=-DIFFERENCE_THRESHOLD, color='r', linestyle='--')
    axs[0].set_xlabel(data['x_axis_label'])
    axs[0].set_ylabel('Difference')
    
    if distribution_a is not None and distribution_b is not None and utilization_fraction_a is not None and utilization_fraction_b is not None:

        distribution_a = distribution_a * utilization_fraction_a
        distribution_b = distribution_b * utilization_fraction_b
        cumsum_distribution_a = np.cumsum(distribution_a)
        cumsum_distribution_b = np.cumsum(distribution_b)
        smooth_cumsum_distribution_a = gaussian_smoothing(cumsum_distribution_a)
        smooth_cumsum_distribution_b = gaussian_smoothing(cumsum_distribution_b) 
        smooth_cumsum_difference_signal = smooth_cumsum_distribution_a - smooth_cumsum_distribution_b

        axs[0].plot(smooth_cumsum_difference_signal, label=f'Smoothed Cumulative Difference In Distributions')

        # shade the area where the Smoothed Cumulative Difference In Distributions is greater than the threshold
        lower_shading_bound = min(min(p_greater_than_threshold_favoring_a), min(p_greather_than_threshold_favoring_b))
        upper_shading_bound = max(max(p_greater_than_threshold_favoring_a), max(p_greather_than_threshold_favoring_b))
        axs[1].fill_between(range(len(smooth_cumsum_difference_signal)), upper_shading_bound, lower_shading_bound, where=smooth_cumsum_difference_signal > DIFFERENCE_THRESHOLD, color='r', alpha=0.3)
        # shade the area where the Smoothed Cumulative Difference In Distributions is less than the negative threshold
        axs[1].fill_between(range(len(smooth_cumsum_difference_signal)), upper_shading_bound, lower_shading_bound, where=smooth_cumsum_difference_signal < -DIFFERENCE_THRESHOLD, color='g', alpha=0.3)

    # add a title
    axs[1].set_title('Probability of Difference Signal Exceeding Threshold')
    
    axs[1].plot(p_greater_than_threshold_favoring_a, label=f'P(Difference Signal > {DIFFERENCE_THRESHOLD}) (Favoring {data["a_name"]})')
    axs[1].plot(p_greather_than_threshold_favoring_b, label=f'P(Difference Signal < -{DIFFERENCE_THRESHOLD}) (Favoring {data["b_name"]})')
    axs[1].axhline(y=CI_P_VALUE, color='r', linestyle='--', label=f'{CI_P_VALUE} Confidence Level')
    axs[1].set_xlabel(data['x_axis_label'])
    axs[1].set_ylabel('Probability')

    axs[0].legend()
    axs[1].legend()
    
def get_smooth_cumsum_difference_signal(distribution_a: np.ndarray, 
                                        distribution_b: np.ndarray,
                                        utilization_fraction_a: float,
                                        utilization_fraction_b: float,
                                        smoothing_fn: Callable) -> np.ndarray:
    distribution_a = distribution_a * utilization_fraction_a
    distribution_b = distribution_b * utilization_fraction_b
    cumsum_distribution_a = np.cumsum(distribution_a)
    cumsum_distribution_b = np.cumsum(distribution_b)
    smooth_cumsum_distribution_a = smoothing_fn(cumsum_distribution_a)
    smooth_cumsum_distribution_b = smoothing_fn(cumsum_distribution_b)
    return smooth_cumsum_distribution_a - smooth_cumsum_distribution_b

def get_fraction_of_days_with_difference_signal_between_confidence_intervals(distribution_a: np.ndarray, 
                                                                             distribution_b: np.ndarray, 
                                                                             ci_lower: np.ndarray, 
                                                                             ci_upper: np.ndarray, 
                                                                             utilization_fraction_a: float, 
                                                                             utilization_fraction_b: float) -> float:
    print('hi')

def generate_data_and_analyze():

    SMOOTHING_KERNEL: Literal['Rolling Average', 'Gaussian', None] = 'Gaussian'
    smoothing_kernel_message = None 

    smoothing_fn = None
    if SMOOTHING_KERNEL == 'Rolling Average':
        def rolling_average(signal: np.ndarray) -> np.ndarray:
            WINDOW_SIZE = 5
            return np.convolve(signal, np.ones(WINDOW_SIZE)/5, mode='same') 
        smoothing_fn = rolling_average
        smoothing_kernel_message = f'with Rolling Average (w={WINDOW_SIZE})'

    elif SMOOTHING_KERNEL == 'Gaussian':
        SIGMA = 3
        TRUNCATE = 9
        def gaussian_smoothing(signal: np.ndarray) -> np.ndarray:
                from scipy.ndimage import gaussian_filter1d
                return gaussian_filter1d(signal, sigma=SIGMA, mode='reflect', truncate=TRUNCATE)
        smoothing_fn = gaussian_smoothing
        smoothing_kernel_message = f'with Gaussian Smoothing (σ={SIGMA})'


    t = np.linspace(0, 200, 200)
    rv_a = stats.gamma(a=2, scale=30)
    rv_b = stats.gamma(a=3, scale=30)
    y_a = rv_a.pdf(-t+200)
    y_b = rv_b.pdf(-t+200)
    # normalize the pdf so that the integral is 1
    y_a /= y_a.sum()
    y_b /= y_b.sum()

    dist_a = y_a
    dist_b = y_b 
    data_generator = UtilizationDataGenerator(
        distribution_a = dist_a,
        distribution_b = dist_b,
        data_range = (0, 200),
        num_bins = 200,
        a_name = 'Group A',
        b_name = 'Group B',
        x_axis_label = 'Days',
        series_name = 'Example Utilization'
    )
    utilization_fraction_a = 0.5
    utilization_fraction_b = 0.5
    data = data_generator.generate_data(12, 20, utilization_fraction_a, utilization_fraction_b)
    analyze_utilization_data(data, dist_a, dist_b, utilization_fraction_a, utilization_fraction_b, smoothing_fn=smoothing_fn, smoothing_kernel_message=smoothing_kernel_message)

generate_data_and_analyze()