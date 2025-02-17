import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_fig1(results, n_replications, sample_size, num_features, alpha, fig_name):


    _fail_if_invalid_inputs(n_replications, sample_size, num_features, alpha, fig_name)

    rfsss = results["rfsss"]
    rfgsss = results["rfgsss"]
    rfsds = results["rfsds"]

    # Histogram settings

    tse1, tse2, se1, se2, sc = _get_standard_errors(rfgsss, rfsds, alpha)

    lb, ub = _get_bounds(tse1, tse2, sc)

    nss, nds, xss, xds = _get_histogram_bin_settings1(tse1, tse2, n_replications)

    bin_width_xss, bin_width_xds = _get_histogram_bin_settings2(xss, xds)

    pss, pds = _get_histogram_bin_settings3(nss, nds, bin_width_xss, bin_width_xds)

    ym = _get_histogram_bin_settings4(se1, se2, lb, ub, pss, pds)   

    if fig_name == "Non-Orthogonal":
        figure = _plot_nonorthogonal_results(xss, pss, lb, ub, se1, sample_size, num_features, ym)
    elif fig_name == "Orthogonal":
        figure = _plot_orthogonal_results(xds, pds, lb, ub, se2, sample_size, num_features, ym)
    elif fig_name == "Overlaid":
        figure = _plot_overlaid_results(rfgsss, rfsss, rfsds) 

    return figure



def _get_standard_errors(rfgsss, rfsds, alpha):
    
    se1 = np.std(rfgsss[:, 0])
    se2 = np.std(rfsds[:, 0])
    sc = max(se1, se2)

    _fail_if_invalid_standard_errors(sc)

    tse1 = se1 * (rfgsss[:, 0] - alpha) / rfgsss[:, 1]
    tse2 = se2 * (rfsds[:, 0] - alpha) / rfsds[:, 1]

    return tse1, tse2, se1, se2, sc

def _get_bounds(tse1, tse2, sc):
    lb = min(np.min(tse1), np.min(tse2)) - 0.25 * sc
    ub = max(np.max(tse1), np.max(tse2)) + 0.25 * sc
    lb = -max(abs(lb), abs(ub))
    ub = max(abs(lb), abs(ub))
    return lb, ub

def _get_histogram_bin_settings1(tse1, tse2, n_replications):
    nss, xss = np.histogram(tse1, bins=int(np.sqrt(n_replications)), density=True)
    nds, xds = np.histogram(tse2, bins=int(np.sqrt(n_replications)), density=True)

    _fail_if_invalid_histogram_bin_counts(nss, nds)

    return nss, nds, xss, xds

def _get_histogram_bin_settings2(xss, xds):

    bin_width_xss = xss[1] - xss[0]
    bin_width_xds = xds[1] - xds[0]

    _fail_if_invalid_histogram_bin_widths(bin_width_xss, bin_width_xds)

    return bin_width_xss, bin_width_xds
    
def _get_histogram_bin_settings3(nss, nds, bin_width_xss, bin_width_xds):

    pss = nss / nss.sum() / bin_width_xss
    pds = nds / nds.sum() / bin_width_xds

    return pss, pds

def _get_histogram_bin_settings4(se1, se2, lb, ub, pss, pds):

    ym = max(np.max(pss), np.max(pds))
    ym = 1.05 * max(
        max(norm.pdf(np.linspace(lb, ub, 1000), 0, se1)),
        max(norm.pdf(np.linspace(lb, ub, 1000), 0, se2)),
        ym,
    )

    return ym

def _plot_nonorthogonal_results(xss, pss, lb, ub, se1, sample_size, num_features, ym):
    # Plot Non-Orthogonal results
    fig, ax = plt.subplots()

    ax.bar(
        (xss[:-1] + xss[1:]) / 2,
        pss,
        width=xss[1] - xss[0],
        color=[0, 0.5, 0.75],
        alpha=0.5,
    )
    ax.plot(np.linspace(lb, ub, 1000), norm.pdf(np.linspace(lb, ub, 1000), 0, se1), "r")
    ax.set_title(f"Non-Orthogonal, n = {sample_size}, p = 5000")
    ax.legend(["Simulation", "N(0, Sigma_s)"], loc="upper left")
    ax.set_xlim([lb, ub])
    ax.set_ylim([0, ym])

    return fig

def _plot_orthogonal_results(xds, pds, lb, ub, se2, sample_size, num_features, ym):
    # Plot Orthogonal results
    fig, ax = plt.subplots()

    ax.bar(
        (xds[:-1] + xds[1:]) / 2,
        pds,
        width=xds[1] - xds[0],
        color=[0, 0.5, 0.75],
        alpha=0.5,
    )
    ax.plot(np.linspace(lb, ub, 1000), norm.pdf(np.linspace(lb, ub, 1000), 0, se2), "r")
    ax.set_title(f"Orthogonal, n = {sample_size}, p = 5000")
    ax.legend(["Simulation", "N(0, Sigma_s)"], loc="upper left")
    ax.set_xlim([lb, ub])
    ax.set_ylim([0, ym])

    return fig

def _plot_overlaid_results(rfgsss, rfsss, rfsds):
    # Overlaid histograms without manually defined edges
    ngsss, edges_ngsss = np.histogram(rfgsss[:, 0], density=True)
    nsss, edges_nsss = np.histogram(rfsss[:, 0], density=True)
    nsds, edges_nsds = np.histogram(rfsds[:, 0], density=True)

    # Compute bin centers and plot
    fig, ax = plt.subplots()

    ax.bar((edges_ngsss[:-1] + edges_ngsss[1:]) / 2, ngsss, width=edges_ngsss[1] - edges_ngsss[0], color=[0, 0, 1], alpha=0.25, label="Non-Orthogonal")
    ax.bar((edges_nsds[:-1] + edges_nsds[1:]) / 2, nsds, width=edges_nsds[1] - edges_nsds[0], color=[1, 0, 0], alpha=0.25, label="Orthogonal")
    ax.axvline(x=0.5, color='k', linestyle='--', label="Reference (0.5)")
    ax.set_title(r'Simulated Distributions of $\widehat{\Theta}$', fontsize=12)
    ax.legend()

    return fig


def _fail_if_invalid_inputs(n_replications, sample_size, num_features, alpha, fig_name):
    """Check if input parameters are valid."""
    if n_replications <= 0 or sample_size <= 0 or num_features <= 0:
        raise ValueError("n_replications, sample_size, and num_features must be positive integers.")
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1.")
    if fig_name not in ["Non-Orthogonal", "Orthogonal", "Overlaid"]:
        raise ValueError("fig_name must be one of 'Non-Orthogonal', 'Orthogonal', or 'Overlaid'.")

def _fail_if_invalid_standard_errors(sc):
    if sc == 0:
        raise ValueError("Standard error cannot be zero. Check your input data.")   

def _fail_if_invalid_histogram_bin_counts(nss,nds):
    if nss.sum() == 0 or nds.sum() == 0:
        raise ValueError(
            "Histogram bin counts cannot be zero. Check your data or binning strategy."
        )
    
def _fail_if_invalid_histogram_bin_widths(bin_width_xss, bin_width_xds):
    if bin_width_xss == 0 or bin_width_xds == 0:
        raise ValueError(
            "Histogram bin width cannot be zero. Check your data or binning strategy."
        )
    











