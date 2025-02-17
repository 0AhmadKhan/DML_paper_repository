"""Tasks for plotting of Figures 1 and 2."""

import pickle

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))  # Ensure the script directory is in Python path

from config import BLD, start_params_fig1, start_params_fig2
from plot_figure1 import plot_fig1
from plot_figure2 import plot_fig2

products = {
    "Non-Orthogonal": BLD / "figures" / "Non-Orthogonal.png",
    "Orthogonal": BLD / "figures" / "Orthogonal.png",
    "Overlaid": BLD / "figures" / "Overlaid.png",
}


# def task_plot_figure1(
#     data_file=BLD / "data" / "figure1.pkl",
#     # script=SRC / "final" / "plot_figure1.py",
#     produces=products,
# ):
#     """Generates and saves plots for Figure 1."""

#     n_replications  = start_params_fig1["n_replications"]
#     sample_size     = start_params_fig1["sample_size"]
#     num_features    = start_params_fig1["num_features"]
#     alpha           = start_params_fig1["alpha"]

#     with Path.open(data_file, "rb") as f:
#         results = pickle.load(f)
#     for fig_name, fig_file in produces.items():
#         fig = plot_fig1(results, n_replications, sample_size, num_features, alpha, fig_name)
#         fig.savefig(fig_file)  


def task_plot_figure2(
    data_file=BLD / "data" / "figure2_for_paper.pkl",
    # script=SRC / "final" / "plot_figure2.py",
    produces=BLD / "figures" / "sample_splitting_paper.png",
):
    """Generates and saves plots for Figure 1."""
    with Path.open(data_file, "rb") as f:
        results = pickle.load(f)
    fig = plot_fig2(results)
    fig.savefig(produces)  











