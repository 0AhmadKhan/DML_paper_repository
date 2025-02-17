# ==================================================================================
# IMPORTS
# ==================================================================================
#region

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from scipy import stats

from data_generator_sparsity import *
# from sim_8_functions import *
# from sim_8_thetas_linear import *
# from sim_8_thetas_non_linear import *

warnings.filterwarnings("ignore")

#endregion



def generate_plots(raw_data, top_heading):

    # Get parameters
    params = true_parameters()
    theta = params["theta"]
    beta = np.array(params["beta"])
    eta = np.array(params["eta"])
    n_samples = params["n_samples"]

    scaled_raw_data = _scale_columns(raw_data, theta, n_samples)

    # Titles for the two plots
    titles = [r'$\hat{\theta}$', r'$\check{\theta}$']
    
    data = [scaled_raw_data.iloc[:, i] for i in range(len(titles))]
    
    # Calculate dynamic x-axis limits based on data and normal distribution range
    all_data = np.concatenate(data)
    data_x_min, data_x_max = all_data.min(), all_data.max()
    normal_x_min, normal_x_max = -4, 4  # Covers most of N(0, 1) (99.7% within ±3σ)
    x_min = min(data_x_min, normal_x_min) - 1  # Add padding of 1 unit
    x_max = max(data_x_max, normal_x_max) + 1  # Add padding of 1 unit

    y_max = 0  # To dynamically calculate the maximum y value

    # Create a single row of plots (1x2 grid)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot each histogram
    for ax, title, estimates in zip(axes, titles, data):
        
        # Plot histogram
        counts, bins, patches = ax.hist(
            estimates, bins=20, color='skyblue', alpha=0.7, edgecolor='black', density=True
        )
        
        # Update y_max dynamically based on the histogram counts
        y_max = max(y_max, counts.max())

        # Plot normal distribution
        x = np.linspace(x_min, x_max, 500)
        pdf = stats.norm.pdf(x, loc=0, scale=1)
        ax.plot(x, pdf, color='red', linestyle='--', linewidth=2, label="N(0, 1)")

        ax.set_title(title)
        ax.set_xlabel('Scaled Theta Estimates')
        ax.set_ylabel('Density')
        ax.legend()

        # Set axis limits dynamically
        ax.set_xlim(x_min, x_max)

    # Apply consistent y-axis limits across both plots
    for ax in axes:
        ax.set_ylim(0, y_max + 0.1 * y_max)  # Add a little padding to y-axis

    # Add a title for the entire figure
    fig.suptitle(top_heading, fontsize=16, fontweight='bold')

    # Adjust layout to make space for the suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig





def _scale_columns(raw_data, theta, n_samples):

    scale_factor = np.sqrt(n_samples)
    
    # Apply transformation to each column
    scaled_data = {
        col: scale_factor * (raw_data[col].to_numpy() - theta)
        for col in raw_data.columns
    }
    
    # Create and return a new DataFrame
    scaled_raw_data = pd.DataFrame(scaled_data)
    return scaled_raw_data
