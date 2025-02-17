"""Tasks for running sims for Figures 1 and 2."""

import pickle

import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True
pd.options.plotting.backend = "plotly"

sys.path.append(str(Path(__file__).resolve().parent))  # Ensure the script directory is in Python path

from figure1 import run_sim_fig1
from figure2 import run_sim_fig2
from config import BLD, start_params_fig1, start_params_fig2 


# def task_run_sim_for_figure1(
#     # script=SRC / "analysis" / "figure1.py",
#     produces=BLD / "data" / "figure1.pkl",
# ):
#     """Conduct the MC simulation for figure 1."""
#     n_replications  = start_params_fig1["n_replications"]
#     sample_size     = start_params_fig1["sample_size"]
#     num_features    = start_params_fig1["num_features"]
#     alpha           = start_params_fig1["alpha"]
#     rho             = start_params_fig1["rho"]
#     seed            = start_params_fig1["seed"]
    
#     data = run_sim_fig1(
#         n_replications=n_replications,
#         sample_size=sample_size,
#         num_features=num_features,
#         alpha=alpha,
#         rho=rho,
#         seed=seed
#     )
#     with Path.open(produces, "wb") as f:
#         pickle.dump(data, f)


def task_run_sim_for_figure2(
    # script=SRC / "analysis" / "figure2.py",
    produces=BLD / "data" / "figure2_for_paper.pkl",
):
    """Conduct the MC simulation for figure 2."""
    num_simulations = start_params_fig2["num_simulations"]
    sample_size     = start_params_fig2["sample_size"]
    seed            = start_params_fig2["seed"]
    
    data = run_sim_fig2(
        num_simulations=num_simulations,
        sample_size=sample_size,
        seed=seed                        
    )
    with Path.open(produces, "wb") as f:
        pickle.dump(data, f)














