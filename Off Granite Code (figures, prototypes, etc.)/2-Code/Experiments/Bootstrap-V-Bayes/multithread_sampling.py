import multiprocessing as mp
from multiprocessing import Manager
from scipy import stats
from tqdm import tqdm
import csv
from functools import partial
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import chi2_contingency, mannwhitneyu, fisher_exact
from typing import TypedDict, Callable, Optional

# CONSTANTS
CI_LEVEL = 0.95
DIFFERENCE_THRESHOLD = 0.05
CI_LEVEL_FOR_DIFFERENCE_THRESHOLD = 0.95
# SMOOTHING_FUNCTION: Callable | None = partial(gaussian_filter1d, sigma=3, truncate=9, mode='nearest')
SMOOTHING_FUNCTION: Callable | None = None 
COMBINATION_RESAMPLE_COUNT = 100

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

def calculate_difference_signal(signals_a: np.ndarray, signals_b: np.ndarray) -> np.ndarray:
    """Calculates the difference signal between two groups. The difference signal is the difference between the utilization signals of the two groups which is the mean of the utilization signals of group A minus the mean of the utilization signals of group B.

    Args:
        tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The first array contains the utilization signals for group A and the second array contains the utilization signals for group B. 

    Returns:
        np.ndarray: A numpy array containing the difference signal between the two groups. The difference signal is the difference between the utilization signals of the two groups.
    """
    return np.mean(signals_a, axis=0) - np.mean(signals_b, axis=0)


def chi_squared_test(n_a: int, n_b: int, a_utilization_fraction: float, b_utilization_fraction: float) -> tuple[float, float]:
    a_positive = int(n_a * a_utilization_fraction)
    a_negative = n_a - a_positive
    b_positive = int(n_b * b_utilization_fraction)
    b_negative = n_b - b_positive

    # check if the contingency table is valid
    if a_positive == 0 or a_negative == 0 or b_positive == 0 or b_negative == 0:
        return None, None
    
    contingency_table = np.array([[a_positive, b_positive],
                                  [a_negative, b_negative]])

    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    return chi2, p_value

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

def get_cumsum_difference_signal(distribution_a: np.ndarray, 
                                        distribution_b: np.ndarray,
                                        utilization_fraction_a: float,
                                        utilization_fraction_b: float,
                                        smoothing_fn: Callable | None = None) -> np.ndarray:
    distribution_a = distribution_a * utilization_fraction_a
    distribution_b = distribution_b * utilization_fraction_b
    cumsum_distribution_a = np.cumsum(distribution_a)
    cumsum_distribution_b = np.cumsum(distribution_b)
    if smoothing_fn is not None:
        smooth_cumsum_distribution_a = smoothing_fn(cumsum_distribution_a)
        smooth_cumsum_distribution_b = smoothing_fn(cumsum_distribution_b)
        return smooth_cumsum_distribution_a - smooth_cumsum_distribution_b
    else:
        return cumsum_distribution_a - cumsum_distribution_b


def daily_fisher_exact(signals_a: np.ndarray,
                      signals_b: np.ndarray):
    # calculate the number of days that each group has a utilization signal of 1
    n_a = signals_a.shape[0]
    n_b = signals_b.shape[0]
    # count the number of days where the utilization signal is 1 for each group (signal not always 0 or 1)
    n_a_positive = np.sum(signals_a == 1, axis=0)
    n_b_positive = np.sum(signals_b == 1, axis=0)

    # perform a chi-squared test on each day and save the p-values 
    statistics = np.zeros(signals_a.shape[1])
    p_values = np.zeros(signals_a.shape[1])
    for i in range(signals_a.shape[1]):
        contingency_table = np.array([[n_a_positive[i], n_b_positive[i]], [n_a - n_a_positive[i], n_b - n_b_positive[i]]])
        statistics[i], p_values[i] = fisher_exact(contingency_table, alternative='two-sided')

    return statistics, p_values
    
    

def analysis_function(combination):
    #print(f'Running analysis for combination {combination}')
    distribution_a_loc = combination[0]
    distribution_b_loc = combination[1] 
    distribution_a_scale = combination[2]
    distribution_b_scale = combination[3]
    a_n = combination[4]
    b_n = combination[5]
    a_utilization_fraction = combination[6]
    b_utilization_fraction = combination[7]

    t = np.arange(0, 200, 1)
    rv_a = stats.norm(loc=distribution_a_loc, scale=distribution_a_scale)
    rv_b = stats.norm(loc=distribution_b_loc, scale=distribution_b_scale)
    y_a = rv_a.pdf(-t+200)
    y_b = rv_b.pdf(-t+200)
    distribution_a = y_a / np.sum(y_a)
    distribution_b = y_b / np.sum(y_b)

    #print(f'Genrating data for combination {combination}')
    data_generator = UtilizationDataGenerator(distribution_a=distribution_a,
                                              distribution_b=distribution_b,
                                              data_range=(0, 200),
                                              num_bins=200,
                                              a_name='A',
                                              b_name='B',
                                              x_axis_label='Time',
                                              series_name='Utilization')

    results = [None] * COMBINATION_RESAMPLE_COUNT 
    for sample_idx in range(COMBINATION_RESAMPLE_COUNT):
        data = data_generator.generate_data(n_a=a_n, 
                                            n_b=b_n, 
                                            utilization_fraction_a=a_utilization_fraction, 
                                            utilization_fraction_b=b_utilization_fraction)


        #print(f"Running boostrap for combination {combination} (resample {sample_idx}/{COMBINATION_RESAMPLE_COUNT})")
        # run boostrap analysis
        signals_a, signals_b = create_utilization_signals(data, smoothing_fn=SMOOTHING_FUNCTION)
        difference_signal = calculate_difference_signal(signals_a, signals_b)
        approximate_sampling_distribution = boostrap_difference_signal_sampling_distribution(signals_a, signals_b)
        ci_lower, ci_upper = get_confidence_intervals_from_sampling_distribution(approximate_sampling_distribution, CI_LEVEL)
        p_greater_than_threshold_favoring_a, p_greather_than_threshold_favoring_b = get_p_difference_signal_greater_than_threshold_in_each_direction(approximate_sampling_distribution, DIFFERENCE_THRESHOLD)

        # evaluate boostrap analysis
        # get fraction of true signal that is within the confidence interval
        difference_signal_from_distributions = get_cumsum_difference_signal(distribution_a=distribution_a,
                                                                                   distribution_b=distribution_b,
                                                                                   utilization_fraction_a=a_utilization_fraction,
                                                                                   utilization_fraction_b=b_utilization_fraction,
                                                                                   smoothing_fn=SMOOTHING_FUNCTION) # the "true" difference signal
        fraction_difference_signal_from_distribution_in_ci = np.sum((difference_signal_from_distributions < ci_upper) & (difference_signal_from_distributions > ci_lower))/len(difference_signal_from_distributions)

        # calculate the sensitivity and specificity of the classification of days as having a difference greater than the difference threshold
        difference_from_distributions_favors_a = difference_signal_from_distributions > DIFFERENCE_THRESHOLD
        difference_from_distributions_favors_b = difference_signal_from_distributions < -DIFFERENCE_THRESHOLD
        difference_from_distributions_exists = difference_from_distributions_favors_a | difference_from_distributions_favors_b

        difference_favoring_a_predicted = p_greater_than_threshold_favoring_a > CI_LEVEL_FOR_DIFFERENCE_THRESHOLD
        difference_favoring_b_predicted = p_greather_than_threshold_favoring_b > CI_LEVEL_FOR_DIFFERENCE_THRESHOLD
        difference_predicted = difference_favoring_a_predicted | difference_favoring_b_predicted
        
        day_diff_true_positive_count = np.sum(difference_from_distributions_exists & difference_predicted)
        day_diff_false_positive_count = np.sum(np.logical_not(difference_from_distributions_exists) & difference_predicted)
        day_diff_true_negative_count = np.sum(np.logical_not(difference_from_distributions_exists) & np.logical_not(difference_predicted))
        day_diff_false_negative_count = np.sum(difference_from_distributions_exists & np.logical_not(difference_predicted))


        if day_diff_true_positive_count + day_diff_false_negative_count != 0:
            day_diff_sensitivity = day_diff_true_positive_count / (day_diff_true_positive_count + day_diff_false_negative_count)
        else:
            day_diff_sensitivity = None

        if day_diff_true_negative_count + day_diff_false_positive_count != 0:
            day_diff_specificity = day_diff_true_negative_count / (day_diff_true_negative_count + day_diff_false_positive_count)
        else: 
            day_diff_specificity = None
        

        # run hypothesis tests
        chi_squared, chi_p_value = chi_squared_test(a_n, b_n, a_utilization_fraction, b_utilization_fraction)
        if data['data_a'].shape[0] != 0 and data['data_b'].shape[0] != 0:
            mann_whitney_statistic, mann_whitney_p_value = mannwhitneyu(data['data_a'], data['data_b'], alternative='two-sided')
        else:
            mann_whitney_statistic = None
            mann_whitney_p_value = None
        
        results[sample_idx] = [*combination, 
              chi_squared, 
              chi_p_value, 
              mann_whitney_statistic, 
              mann_whitney_p_value, 
              fraction_difference_signal_from_distribution_in_ci, 
              day_diff_sensitivity,
              day_diff_specificity]

    return results


def process_batch(batch, output_file, lock):
    results = [analysis_function(combination) for combination in batch]
    with lock:
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for result in results:
                #print("RESULT IS:")
                #print(result)
                writer.writerows(result)

def parallel_processing_with_batched_writing(sampled_combinations, output_file, sample_spaces_name, batch_size=100, num_processes=None):
    if num_processes is None:
        num_processes = mp.cpu_count()

        # leave 2 cores for other processes so that the system doesn't hang
        if num_processes > 4:
            num_processes -= 2
    
    print(f"Using {num_processes} processes")

    with Manager() as manager:
        lock = manager.Lock()
        process_func = partial(process_batch, output_file=output_file, lock=lock)

        open(output_file, 'w').close()

        with lock:
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(sample_spaces_name + ['chi_squared', 'chi_p_value', 'mann_whitney_statistic', 'mann_whitney_p_value', 'fraction_difference_signal_from_distribution_in_ci', 'day_diff_sensitivity', 'day_diff_specific'])

        with mp.Pool(processes=num_processes) as pool:
            batches = [sampled_combinations[i:i+batch_size] for i in range(0, len(sampled_combinations), batch_size)]
            list(tqdm(
                pool.imap(process_func, batches),
                total=len(batches),
                desc="Processing batches"
            ))

def cross_product_sampling(sample_spaces, num_samples=1000):
    sampled_combinations = [sample_space[np.random.choice(len(sample_space), num_samples, replace=True)] for sample_space in sample_spaces]
    return [[sampled_combinations[j][i] for j in range(len(sample_spaces))] for i in range(num_samples)]

def cps_sanity_check():
    sample_spaces = [np.array([False, True]), np.array([10,20,30,40,50,60])]
    sampled_combinations = cross_product_sampling(sample_spaces, num_samples=100)
    print(sampled_combinations)


############################################# MAIN #############################################
############################################# MAIN #############################################
############################################# MAIN #############################################

if __name__ == '__main__':
    action = 'sanity_check'
    if action == 'validation':
        output_file = 'results_3.csv'
    
        # define sample spaces
        distribution_a_loc = np.linspace(0, 200, 400)
        distribution_b_loc = np.linspace(0, 200, 400)
        distribution_a_scale = np.linspace(1, 100, 200)
        distribution_b_scale = np.linspace(1, 100, 200)
        a_n = np.arange(5, 100, 1)
        b_n = np.arange(5, 100, 1)
        a_utilization_fraction = np.linspace(0.1, 0.9, 20)
        b_utilization_fraction = np.linspace(0.1, 0.9, 20)

        sample_spaces_names = ['distribution_a_loc', 'distribution_b_loc', 'distribution_a_scale', 'distribution_b_scale', 'a_n', 'b_n', 'a_utilization_fraction', 'b_utilization_fraction']

        sample_spaces = [distribution_a_loc, distribution_b_loc, distribution_a_scale, distribution_b_scale, a_n, b_n, a_utilization_fraction, b_utilization_fraction] 
        sampled_combinations = cross_product_sampling(sample_spaces, num_samples=100000)
        parallel_processing_with_batched_writing(sampled_combinations, output_file, sample_spaces_names, batch_size=50)
        print(f'Output written to {output_file}')
    elif action == 'sanity_check':
        # generate some sample data
        distribution_a_loc = 100
        distribution_b_loc = 199
        distribution_a_scale = 20
        distribution_b_scale = 100
        a_n = 30
        b_n = 30
        a_utilization_fraction = .9
        b_utilization_fraction = .75

        t = np.arange(0, 200, 1)
        rv_a = stats.norm(loc=distribution_a_loc, scale=distribution_a_scale)
        rv_b = stats.norm(loc=distribution_b_loc, scale=distribution_b_scale)
        y_a = rv_a.pdf(-t+200)
        y_b = rv_b.pdf(-t+200)
        distribution_a = y_a / np.sum(y_a)
        distribution_b = y_b / np.sum(y_b)

        #print(f'Genrating data for combination {combination}')
        data_generator = UtilizationDataGenerator(distribution_a=distribution_a,
                                                  distribution_b=distribution_b,
                                                  data_range=(0, 200),
                                                  num_bins=200,
                                                  a_name='A',
                                                  b_name='B',
                                                  x_axis_label='Time',
                                                  series_name='Utilization') 
    
        data = data_generator.generate_data(n_a=a_n, 
                                            n_b=b_n, 
                                            utilization_fraction_a=a_utilization_fraction, 
                                            utilization_fraction_b=b_utilization_fraction)
        signals_a, signals_b = create_utilization_signals(data, smoothing_fn=SMOOTHING_FUNCTION)
        difference_signal = calculate_difference_signal(signals_a, signals_b)
        approximate_sampling_distribution = boostrap_difference_signal_sampling_distribution(signals_a, signals_b)
        ci_lower, ci_upper = get_confidence_intervals_from_sampling_distribution(approximate_sampling_distribution, CI_LEVEL)
        p_greater_than_threshold_favoring_a, p_greather_than_threshold_favoring_b = get_p_difference_signal_greater_than_threshold_in_each_direction(approximate_sampling_distribution, DIFFERENCE_THRESHOLD)

        # evaluate boostrap analysis
        # get fraction of true signal that is within the confidence interval
        difference_signal_from_distributions = get_cumsum_difference_signal(distribution_a=distribution_a,
                                                                                   distribution_b=distribution_b,
                                                                                   utilization_fraction_a=a_utilization_fraction,
                                                                                   utilization_fraction_b=b_utilization_fraction,
                                                                                   smoothing_fn=SMOOTHING_FUNCTION) # the "true" difference signal
        fraction_difference_signal_from_distribution_in_ci = np.sum((difference_signal_from_distributions < ci_upper) & (difference_signal_from_distributions > ci_lower))/len(difference_signal_from_distributions)

        # calculate the sensitivity and specificity of the classification of days as having a difference greater than the difference threshold
        difference_from_distributions_favors_a = difference_signal_from_distributions > DIFFERENCE_THRESHOLD
        difference_from_distributions_favors_b = difference_signal_from_distributions < -DIFFERENCE_THRESHOLD
        difference_from_distributions_exists = difference_from_distributions_favors_a | difference_from_distributions_favors_b

        difference_favoring_a_predicted = p_greater_than_threshold_favoring_a > CI_LEVEL_FOR_DIFFERENCE_THRESHOLD
        difference_favoring_b_predicted = p_greather_than_threshold_favoring_b > CI_LEVEL_FOR_DIFFERENCE_THRESHOLD
        difference_predicted = difference_favoring_a_predicted | difference_favoring_b_predicted
        
        day_diff_true_positive_count = np.sum(difference_from_distributions_exists & difference_predicted)
        day_diff_false_positive_count = np.sum(np.logical_not(difference_from_distributions_exists) & difference_predicted)
        day_diff_true_negative_count = np.sum(np.logical_not(difference_from_distributions_exists) & np.logical_not(difference_predicted))
        day_diff_false_negative_count = np.sum(difference_from_distributions_exists & np.logical_not(difference_predicted))


        # sensitivity is none if there are neither true positives nor false negatives (to avoid division by zero)
        if day_diff_true_positive_count + day_diff_false_negative_count != 0:
            day_diff_sensitivity = day_diff_true_positive_count / (day_diff_true_positive_count + day_diff_false_negative_count)
        else:
            day_diff_sensitivity = None

        # specificity is none if there are neither true negatives nor false positives (to avoid division by zero)
        if day_diff_true_negative_count + day_diff_false_positive_count != 0:
            day_diff_specificity = day_diff_true_negative_count / (day_diff_true_negative_count + day_diff_false_positive_count)
        else: 
            day_diff_specificity = None
        

        # run hypothesis tests
        fe_statistics, fe_p_values = daily_fisher_exact(signals_a, signals_b)

        if data['data_a'].shape[0] != 0 and data['data_b'].shape[0] != 0:
            mann_whitney_statistic, mann_whitney_p_value = mannwhitneyu(data['data_a'], data['data_b'], alternative='two-sided')
        else:
            mann_whitney_statistic = None
            mann_whitney_p_value = None

        # plot the fe_p_values and the difference signal with the confidence intervals
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        ax[0].plot(fe_p_values)
        ax[0].plot(1 - p_greater_than_threshold_favoring_a)
        ax[0].plot(1 - p_greather_than_threshold_favoring_b)
        ax[0].legend(['Fisher Exact P-Values', 'P(Difference Signal > 0.05) for A', 'P(Difference Signal < -0.05) for B'])
        ax[0].axhline(0.05, color='r', linestyle='--')
        ax[0].set_title('Fisher Exact P-Values')
        ax[1].plot(difference_signal)
        ax[1].plot(difference_signal_from_distributions)
        ax[1].fill_between(np.arange(0, 200), ci_lower, ci_upper, color='gray', alpha=0.5)
        ax[1].set_title('Difference Signal with Confidence Intervals')

        plt.show()
        
        results = [
              mann_whitney_statistic, 
              mann_whitney_p_value, 
              fraction_difference_signal_from_distribution_in_ci, 
              day_diff_sensitivity,
              day_diff_specificity]
        
        print(results)

