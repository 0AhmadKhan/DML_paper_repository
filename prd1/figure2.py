import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def run_sim_fig2(num_simulations, sample_size, seed):

    rng = np.random.default_rng(seed)

    full_sample_estimates = np.zeros(num_simulations)
    split_sample_estimates = np.zeros(num_simulations)

    for sim in range(num_simulations):
        half_sample = rng.standard_normal(sample_size // 2)
        z = np.concatenate((half_sample, half_sample))
        u = rng.standard_normal(sample_size)
        v = rng.standard_normal(sample_size)
        x = z + u
        y = x + z + v

        model_estimate = z
        overfit_correction = (y - z) / (sample_size ** (1 / 3))

        full_sample_estimates[sim] = np.dot(
            (x - model_estimate).T, (y - z - overfit_correction)
        ) / np.dot((x - model_estimate).T, x)

        split_a_indices = np.arange(sample_size // 2)
        split_b_indices = np.arange(sample_size // 2, sample_size)

        estimate_a = np.dot(
            u[split_a_indices].T,
            (y[split_a_indices] - z[split_a_indices] - overfit_correction[split_b_indices]),
        ) / np.dot(u[split_a_indices].T, x[split_a_indices])
        estimate_b = np.dot(
            u[split_b_indices].T,
            (y[split_b_indices] - z[split_b_indices] - overfit_correction[split_a_indices]),
        ) / np.dot(u[split_b_indices].T, x[split_b_indices])

        split_sample_estimates[sim] = 0.5 * (estimate_a + estimate_b)

    results = {
        "full_sample_estimates" : full_sample_estimates,
        "split_sample_estimates": split_sample_estimates
    }

    return results
