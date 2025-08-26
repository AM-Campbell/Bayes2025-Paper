import matplotlib.pyplot as plt
import numpy as np
from campbell_color_utils import get_longitudinal_care_delivery_paper_colormap
from termcolor import colored

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern font for math
plt.style.use('seaborn-v0_8-colorblind')

print("plotting functions loaded")


def pcd_hist_dep(fig, ax, data, **kwargs):
    """
    Create a stacked bar chart of two distributions with markers for the third distribution
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    data : dict
        Dictionary containing the distributions
    **kwargs : dict
        Additional keyword arguments for customization
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The modified figure object
    """
    dist1 = data['distribution_of_wmpcdp_days']
    dist2 = data['distribution_of_pmpcdp_days']
    dist3 = data['distribution_of_dbdfep_days']
    
    # Get color scheme
    colors = get_longitudinal_care_delivery_paper_colormap()
    
    # Verify all distributions have the same length
    n_points = len(dist1)
    assert len(dist2) == n_points, "Distributions must have equal length"
    assert len(dist3) == n_points, "Distributions must have equal length"
    
    # Create x-coordinates for bars
    x = np.arange(n_points)

    x = x + 1
    
    # Create stacked bar chart
    ax.bar(x, dist1, 
        #    label='WMPCDP Days',
           color=colors['wmpcd'],
           alpha=0.75)
    
    ax.bar(x, dist2,
           bottom=dist1,  # Stack on top of first distribution
        #    label='PMPCD Days',
           color=colors['pmpcd'],
           alpha=0.75)
    
    # Plot just markers for third distribution
    ax.scatter(x, dist3,
            #   label='DBDFEP Days',
              color=colors['dbdfe'],
              marker='o',
              s=40)  # Size of markers
    
    # Customize the plot
    ax.set_title('Distribution of Positive Days')
    ax.set_xlabel('Day')
    ax.set_ylabel('Count of Positive Days')
    #ax.legend()
    
    # Set x-axis limits
    ax.set_xlim(-0.5, n_points-0.5)

    ax.grid(True, alpha=1, linewidth=0.2, color='black')
    # Remove the top and right spines (bounding lines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig

def plot_two(fig, ax, data, **kwargs):
    """Create a scatter plot"""
    x = data['data']['x']
    y = data['data']['y2']
    
    ax.scatter(x, y, c='red', s=100)
    ax.set_title('Simple Scatter Plot')
    ax.set_xlabel('X values')
    ax.set_ylabel('Y values')
    ax.grid(True)
    
    return fig

def pcd_hist(fig, ax, data, **kwargs):
    """
    Create a stacked area chart of two distributions with markers for the third distribution
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    data : dict
        Dictionary containing the distributions
    **kwargs : dict
        Additional keyword arguments for customization
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The modified figure object
    """
    dist1 = data['distribution_of_wmpcdp_days']
    dist2 = data['distribution_of_pmpcdp_days']
    dist3 = data['distribution_of_dbdfep_days']
    
    # Get color scheme
    colors = get_longitudinal_care_delivery_paper_colormap()
    
    # Verify all distributions have the same length
    n_points = len(dist1)
    assert len(dist2) == n_points, "Distributions must have equal length"
    assert len(dist3) == n_points, "Distributions must have equal length"
    
    # Create x-coordinates
    x = np.arange(n_points)
    x = x + 1
    
    # Create stacked area chart
    ax.stackplot(x, [dist1, dist2],
                labels=[r'$\mathbf{WMPCD}(+)$', r'$\mathbf{PMPCD}(+)$'],
                colors=[colors['wmpcd'], colors['pmpcd']],
                alpha=0.75)
    
    # Plot just markers for third distribution
    ax.scatter(x, dist3,
              label=r'$\mathbf{DBDFE}(+)$',
              color=colors['dbdfe'],
              marker='o',
              s=40)  # Size of markers
    
    # Customize the plot
    ax.set_title('Distribution of Positive Days')
    ax.set_xlabel('Day')
    ax.set_ylabel('Count of Positive Days')
    #ax.legend()
    
    # Set x-axis limits
    ax.set_xlim(1, n_points)
    ax.grid(True, alpha=1, linewidth=0.2, color='black')
    # Remove the top and right spines (bounding lines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig

def diff_signal(fig, ax, data, **kwargs):
    print(colored('Generating diff signal plot', 'green'))

    # we're going to make this plot a bit more complicated. It's going to be a two by two grid. 

    colors = get_longitudinal_care_delivery_paper_colormap()

    x = data['day_ind_on_0_199']
    
    # Plot main signal line
    ax.plot(x, data['diffSignal'], 
            color=colors['diffSignal'], 
            linewidth=2, 
            label='Difference Signal')
    
    # plot the white and poc utilizations
    # ax.plot(x, data['w_util'],
    #         color=colors['w_util'],
    #         linewidth=2,
    #         label='White Utilization')
        
    # ax.plot(x, data['p_util'],
    #         color=colors['p_util'],
    #         linewidth=2,
    #         label='POC Utilization')

    # Plot credible interval
    ax.fill_between(x, 
                    data['cilow'], 
                    data['cihigh'], 
                    color=colors['ci'], 
                    alpha=0.2, 
                    label='95% Credible Interval')
    
    # Create mask for threshold regions
    threshold_mask = [(w > 0.95 or p > 0.95) for w, p in zip(data['wmpcd'], data['pmpcd'])]
    
    # Plot threshold regions
    # for i in range(len(x)):
    #     if threshold_mask[i]:
    #         ax.fill_between([x[i], x[i]+1], 
    #                       [-1, -1], 
    #                       [1, 1], 
    #                       color=colors['pcd'], 
    #                       alpha=0.2)
    plot_threshold_regions(ax, x, threshold_mask, color=colors['pcd'], alpha=0.2)
    
    # Add threshold region to legend with custom patch
    from matplotlib.patches import Patch
    threshold_patch = Patch(color=colors['pcd'], 
                          alpha=0.2, 
                          label='WMPCD > 0.95 or\nPMPCD > 0.95')
    
    # Customize plot
    # ax.set_xlabel('Day')
    # ax.set_ylabel('diffSignal')
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=1, linewidth=0.2, color='black')
    # Remove the top and right spines (bounding lines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    # handles, labels = ax.get_legend_handles_labels()
    # handles.append(threshold_patch)
    # ax.legend(handles=handles, loc='upper right') 

    print(colored('returning fig', 'green'))
    
    return fig

    

def plot_threshold_regions(ax, x, mask, color, alpha=0.2):
    in_threshold = False
    start = None

    for i in range(len(x) - 1):
        if mask[i] and not in_threshold:
            # Start of a threshold region
            start = x[i]
            in_threshold = True
        elif not mask[i] and in_threshold:
            # End of a threshold region
            ax.fill_between([start, x[i]], [0, 0], [1, 1], color=color, alpha=alpha)
            in_threshold = False

    # Fill the last segment if it ends in a threshold region
    if in_threshold:
        ax.fill_between([start, x[-1]], [0, 0], [1, 1], color=color, alpha=alpha)

def diff_signal_stats_plot(fig, ax, data, **kwargs):

    colors = get_longitudinal_care_delivery_paper_colormap()

    # Get x-axis data
    x = data['day_ind_on_0_199']
    
    # Plot main lines
    ax.plot(x, data['dbdfe'], 
            color=colors['dbdfe'], 
            linewidth=2, 
            label='DBDFE')
    
    ax.plot(x, data['wmpcd'], 
            color=colors['wmpcd'], 
            linewidth=2, 
            label='WMPCD')
    
    ax.plot(x, data['pmpcd'], 
            color=colors['pmpcd'], 
            linewidth=2, 
            label='PMPCD')
    
    # Create masks for threshold regions
    pcd_threshold_mask = [(w > 0.95 or p > 0.95) for w, p in zip(data['wmpcd'], data['pmpcd'])]
    dbdfe_threshold_mask = [d < 0.05 for d in data['dbdfe']]
    

    # Plot WMPCD/PMPCD threshold regions
    plot_threshold_regions(ax, x, pcd_threshold_mask, color=colors['pcd'], alpha=0.2)

    # Plot DBDFE threshold regions
    plot_threshold_regions(ax, x, dbdfe_threshold_mask, color=colors['dbdfe'], alpha=0.2) 

    # Add threshold regions to legend with custom patches
    from matplotlib.patches import Patch
    wmpm_patch = Patch(color=colors['pcd'], 
                      alpha=0.2, 
                      label='WMPCD > 0.95 or\nPMPCD > 0.95')
    dbdfe_patch = Patch(color=colors['dbdfe'], 
                       alpha=0.2, 
                       label='DBDFE < 0.05')
    
    # Customize plot
    ax.set_xlabel('Day Index')
    # ax.set_ylabel('Metric Value')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=1, linewidth=0.2, color='black')
    
    # Add legend
    # handles, labels = ax.get_legend_handles_labels()
    # handles.extend([wmpm_patch, dbdfe_patch])
    # ax.legend(handles=handles, loc='upper right')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # add a large marker to the final value of dbdfe
    # ax.plot(x[-1], data['dbdfe'][-1], 'o', color=colors['fe'], markersize=10)
    
    return fig, ax

def pcd_pie(fig, ax, data, **kwargs):
    """
    Create a nested pie chart (using a bar chart on polar coordinates) to display the distribution of PCD(+) days between PMPCD and WMPCD.
    Then for each of these show the breakdown between FE(+) and FE(-) hospitals. 

    The series we need to use are 
    """
    print('generating pie chart')   
    
    figsize = fig.get_size_inches()
    # Clear the existing figure
    fig.clear()
    fig.set_size_inches(figsize)
    
    # Create new polar axes in the existing figure
    ax = fig.add_subplot(111, projection='polar')

    size = 0.3 

    values = [[data['fep_wmpcdp'], data['fen_wmpcdp']], 
              [data['fep_pmpcdp'], data['fen_pmpcdp']]]
    
    # Rest of your code remains the same...
    # labels 
    outer_labels = [r'$\mathbf{WMPCD}(+)$', r'$\mathbf{PMPCD}(+)$']
    inner_labels = [r'$\mathbf{FE}(+)$', r'$\mathbf{FE}(-)$']

    # colors
    colors = get_longitudinal_care_delivery_paper_colormap()
    outer_colors = [colors['wmpcd'], colors['pmpcd']]
    inner_colors = [colors['fe'], colors['gray']]

    # normalize the values to 2 pi
    values = np.array(values)
    valsnorm = values / values.sum() * 2 * np.pi
    
    # obtain the coordinates of the bars
    valsleft = np.cumsum(np.append(0, valsnorm.flatten()[:-1])).reshape(values.shape)

    # Plot outer ring
    outer_bars = ax.bar(x=valsleft[:, 0],
                        width=valsnorm.sum(axis=1), bottom=1 - size, height=size,
                        color=outer_colors, edgecolor='w', linewidth=1, align="edge")

    # Plot inner ring
    inner_bars = ax.bar(x=valsleft.flatten(),
                        width=valsnorm.flatten(), bottom=1 - 2 * size, height=size,
                        color=inner_colors, edgecolor='w', linewidth=1, align="edge")

    # Function to add value labels
    def add_labels(bars, values, ring):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            rho = 1 - size / 2 if ring == 'outer' else 1 - 3 * size / 2
            theta = bar.get_x() + bar.get_width() / 2
            x = theta
            y = rho
            ha = 'center'
            va = 'center'
            ax.text(x, y, f'{value:.1f}', ha=ha, va=va, fontweight='bold')

    # Add value labels
    outer_values = values.sum(axis=1)
    inner_values = values.flatten()

    add_labels(outer_bars, outer_values, 'outer')
    add_labels(inner_bars, inner_values, 'inner')

    ax.set_axis_off()

    # Add legend
    outer_legend = ax.legend(outer_bars, outer_labels, loc='upper left', bbox_to_anchor=(1, 1))
    # outer_legend = ax.legend(outer_bars, outer_labels, loc='upper left')
    ax.add_artist(outer_legend)
    ax.legend(inner_bars, inner_labels, loc='upper left', bbox_to_anchor=(1.1, 0.8))
    # ax.legend(inner_bars, inner_labels, loc='upper left')

    print('done generating pie chart')

    return fig, ax

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
################################################################################################################
    
    
    
    
def combined_plot(fig, ax, data, **kwargs):
    print(colored('Generating combined plot', 'green'))
    colors = get_longitudinal_care_delivery_paper_colormap()

    # Get x-axis data
    x = data['day_ind_on_0_199']
    
    # Plot main lines
    ax.plot(x, data['dbdfe'], 
            color=colors['dbdfe'], 
            linewidth=2, 
            label='DBDFE')
    
    ax.plot(x, data['wmpcd'], 
            color=colors['wmpcd'], 
            linewidth=2, 
            label='WMPCD')
    
    ax.plot(x, data['pmpcd'], 
            color=colors['pmpcd'], 
            linewidth=2, 
            label='PMPCD')
    
    # Create masks for threshold regions
    pcd_threshold_mask = [(w > 0.95 or p > 0.95) for w, p in zip(data['wmpcd'], data['pmpcd'])]
    dbdfe_threshold_mask = [d < 0.05 for d in data['dbdfe']]
    

    # Plot WMPCD/PMPCD threshold regions
    plot_threshold_regions(ax, x, pcd_threshold_mask, color=colors['pcd'], alpha=0.2)

    # Plot DBDFE threshold regions
    plot_threshold_regions(ax, x, dbdfe_threshold_mask, color=colors['dbdfe'], alpha=0.2) 

    # Add threshold regions to legend with custom patches
    from matplotlib.patches import Patch
    wmpm_patch = Patch(color=colors['pcd'], 
                      alpha=0.2, 
                      label='WMPCD > 0.95 or\nPMPCD > 0.95')
    dbdfe_patch = Patch(color=colors['dbdfe'], 
                       alpha=0.2, 
                       label='DBDFE < 0.05')
    
    # Customize plot
    ax.set_xlabel('Day Index')
    # ax.set_ylabel('Metric Value')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=1, linewidth=0.2, color='black')
    
    # Add legend
    # handles, labels = ax.get_legend_handles_labels()
    # handles.extend([wmpm_patch, dbdfe_patch])
    # ax.legend(handles=handles, loc='upper right')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # add a large marker to the final value of dbdfe
    # ax.plot(x[-1], data['dbdfe'][-1], 'o', color=colors['fe'], markersize=10)
    
    return fig, ax