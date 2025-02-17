import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_fig2(results):

    full_sample_estimates = results["full_sample_estimates"]
    split_sample_estimates = results["split_sample_estimates"]
   
    edges, xb1, xb2, nb1, nb2 = _get_histogram_bin_settings(full_sample_estimates, split_sample_estimates)

    figure = _plot_sample_splitting_results(xb1, nb1, xb2, nb2, edges)

    return figure


def _get_histogram_bin_settings(full_sample_estimates, split_sample_estimates):
    # Histogram settings
    edges = np.arange(-7, 7.05, 0.05)
    nb1, xb1 = np.histogram(
        (full_sample_estimates - 1) / np.std(full_sample_estimates),
        bins=edges,
        density=True,
    )
    nb2, xb2 = np.histogram(
        (split_sample_estimates - 1) / np.std(split_sample_estimates),
        bins=edges,
        density=True,
    )
    return edges, xb1, xb2, nb1, nb2

def _plot_sample_splitting_results(xb1, nb1, xb2, nb2, edges):
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Full Sample
    axes[0].bar((xb1[:-1] + xb1[1:]) / 2, nb1, width=0.05, color=[0, 0.5, 0.75], alpha=0.5)
    axes[0].plot(edges, norm.pdf(edges), "r")
    axes[0].set_title("A. Full Sample n=500 p=5000")
    axes[0].legend(["Simulation", "N(0,1)"], loc="upper left")

    # Split Sample
    axes[1].bar((xb2[:-1] + xb2[1:]) / 2, nb2, width=0.05, color=[0, 0.5, 0.75], alpha=0.5)
    axes[1].plot(edges, norm.pdf(edges), "r")
    axes[1].set_title("B. Split Sample n=500 p=5000")
    axes[1].legend(["Simulation", "N(0,1)"], loc="upper left")

    return fig




