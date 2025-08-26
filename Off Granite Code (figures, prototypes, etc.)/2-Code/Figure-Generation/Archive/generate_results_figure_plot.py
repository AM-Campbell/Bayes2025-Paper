import matplotlib.pyplot as plt
import numpy as np

# Assume this loads data from the directory, returns dictionary of series (testing mode here)
def load_hospital_results_series(directory, testing=False):
    series_names = [
        'diffSignal',
        'cil',
        'ciu',
        'wmpcd',
        'pmpcd',
        'dbdfe'
    ]

    stats_file_name = 'other_stats.json'  # maybe will not use this

    # For testing, return random numpy arrays
    if testing:
        series = {}
        for series_name in series_names:
            series[series_name] = np.random.rand(200)  # for testing
        return series
    else:
        raise NotImplementedError("This function is not implemented yet.")

def generate_results_figure_plots(path_to_series):
    
    series = load_hospital_results_series(path_to_series, testing=True)  # testing mode
    
    # Extract individual series from the dictionary
    x = np.arange(200)  # Assuming the x-axis is just index values from 0 to 199
    diffSignal = series['diffSignal']
    cil = series['cil']
    ciu = series['ciu']
    wmpcd = series['wmpcd']
    pmpcd = series['pmpcd']
    dbdfe = series['dbdfe']

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the diffSignal line
    ax.plot(x, diffSignal, label='diffSignal', color='blue')

    # Fill between for confidence intervals
    ax.fill_between(x, cil, ciu, color='lightblue', alpha=0.5, label='Confidence Interval')

    # Shaded regions where conditions on wmpcd, pmpcd, and dbdfe hold true
    ax.fill_between(x, -1, 1, where=(wmpcd > 0.95), color='lightgreen', alpha=0.3, label='wmpcd > 0.95')
    ax.fill_between(x, -1, 1, where=(pmpcd > 0.95), color='lightcoral', alpha=0.3, label='pmpcd > 0.95')
    ax.fill_between(x, -1, 1, where=(dbdfe < 0.05), color='lightyellow', alpha=0.3, label='dbdfe < 0.05')

    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Hospital Results: diffSignal with Confidence Intervals')
    ax.set_ylim([-1, 1])

    # Add legend
    ax.legend()

    # Show the plot
    # plt.show()

    # Save the plot
    fig.savefig('hospital_results_diffSignal_plot.png')

# Call the function to generate the plot
 
generate_results_figure_plots("path")

    
    

    