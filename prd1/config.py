"""All the general configuration of the project."""

from pathlib import Path

ROOT = Path(__file__).parent.resolve()
# ROOT = SRC.joinpath("..", "..").resolve()

BLD = ROOT.joinpath("bld").resolve()


# DOCUMENTS = ROOT.joinpath("documents").resolve()

start_params_fig1 = {
    "n_replications": 5,
    "sample_size": 500,
    "num_features": 20,
    "alpha": 0.5,
    "rho": 0.7,
    "seed": 2162016,
}

start_params_fig2 = {"num_simulations": 5000, "sample_size": 500, "seed": 13867}
