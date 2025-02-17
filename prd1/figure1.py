import numpy as np
import scipy.linalg as la
from sklearn.ensemble import RandomForestRegressor


def run_sim_fig1(n_replications, sample_size, num_features, alpha, rho, seed):
    _fail_if_invalid_inputs(n_replications, sample_size, num_features, alpha, rho, seed)

    cholesky_matrix = _create_toeplitz_cholesky(num_features, rho)

    rng = np.random.default_rng(seed)

    # Key:
    # rf   is random forests
    # sss  is (TBF)
    # gsss is (TBF)
    # sds  is (TBF)

    rfsss1 = np.zeros((n_replications, 2))
    rfgsss1 = np.zeros((n_replications, 2))
    rfsds1 = np.zeros((n_replications, 2))
    rfsss2 = np.zeros((n_replications, 2))
    rfgsss2 = np.zeros((n_replications, 2))
    rfsds2 = np.zeros((n_replications, 2))
    rfsss = np.zeros((n_replications, 2))
    rfgsss = np.zeros((n_replications, 2))
    rfsds = np.zeros((n_replications, 2))

    for rep in range(n_replications):
        x = rng.standard_normal((sample_size, num_features)) @ cholesky_matrix

        a0, a1, s1 = 1, 0.25, 1
        b0, b1, s2 = 1, 0.25, 1

        d = (
            a0 * x[:, 0]
            + a1 * np.exp(x[:, 2]) / (1 + np.exp(x[:, 2]))
            + s1 * rng.standard_normal(sample_size)
        )
        y = (
            alpha * d
            + b0 * np.exp(x[:, 0]) / (1 + np.exp(x[:, 0]))
            + b1 * x[:, 2]
            + s2 * rng.standard_normal(sample_size)
        )

        samp = rng.choice(sample_size, size=sample_size // 2, replace=False)
        osamp = np.setdiff1d(np.arange(sample_size), samp)

        # Use first split
        fy1 = RandomForestRegressor(n_estimators=500)
        fy1.fit(x[samp, :], y[samp])
        yhat1 = fy1.predict(x[osamp, :])
        residual_y1 = y[osamp] - yhat1

        fd1 = RandomForestRegressor(n_estimators=500)
        fd1.fit(x[samp, :], d[samp])
        dhat1 = fd1.predict(x[osamp, :])
        residual_d1 = d[osamp] - dhat1

        rfsss1[rep, 0] = np.dot(residual_d1, y[osamp]) / np.dot(residual_d1, d[osamp])
        rfsss1[rep, 1] = np.sqrt(
            np.mean((y[osamp] - d[osamp] * rfsss1[rep, 0]) ** 2)
            * np.dot(residual_d1, residual_d1)
            / (np.dot(residual_d1, d[osamp]) ** 2)
        )

        ghat1 = yhat1 - dhat1 * rfsss1[rep, 0]
        gy1 = y[osamp] - ghat1
        rfgsss1[rep, 0] = np.linalg.lstsq(d[osamp, np.newaxis], gy1, rcond=None)[0][0]

        xpxinv = 1 / np.dot(d[osamp], d[osamp])
        rfgsss1[rep, 1] = _iid_se(
            d[osamp, np.newaxis], gy1 - d[osamp] * rfgsss1[rep, 0], xpxinv
        ).item()

        rfsds1[rep, 0] = np.linalg.lstsq(
            residual_d1[:, np.newaxis], residual_y1, rcond=None
        )[0][0]
        rfsds1[rep, 1] = _iid_se(
            residual_d1[:, np.newaxis],
            residual_y1 - residual_d1 * rfsds1[rep, 0],
            1 / np.dot(residual_d1, residual_d1),
        ).item()

        # Use second split
        fy2 = RandomForestRegressor(n_estimators=500)
        fy2.fit(x[osamp, :], y[osamp])
        yhat2 = fy2.predict(x[samp, :])
        residual_y2 = y[samp] - yhat2

        fd2 = RandomForestRegressor(n_estimators=500)
        fd2.fit(x[osamp, :], d[osamp])
        dhat2 = fd2.predict(x[samp, :])
        residual_d2 = d[samp] - dhat2

        rfsss2[rep, 0] = np.dot(residual_d2, y[samp]) / np.dot(residual_d2, d[samp])
        rfsss2[rep, 1] = np.sqrt(
            np.mean((y[samp] - d[samp] * rfsss2[rep, 0]) ** 2)
            * np.dot(residual_d2, residual_d2)
            / (np.dot(residual_d2, d[samp]) ** 2)
        )

        ghat2 = yhat2 - dhat2 * rfsss2[rep, 0]
        gy2 = y[samp] - ghat2
        rfgsss2[rep, 0] = np.linalg.lstsq(d[samp, np.newaxis], gy2, rcond=None)[0][0]

        xpxinv2 = 1 / np.dot(d[samp], d[samp])
        rfgsss2[rep, 1] = _iid_se(
            d[samp, np.newaxis], gy2 - d[samp] * rfgsss2[rep, 0], xpxinv2
        ).item()

        rfsds2[rep, 0] = np.linalg.lstsq(
            residual_d2[:, np.newaxis], residual_y2, rcond=None
        )[0][0]
        rfsds2[rep, 1] = _iid_se(
            residual_d2[:, np.newaxis],
            residual_y2 - residual_d2 * rfsds2[rep, 0],
            1 / np.dot(residual_d2, residual_d2),
        ).item()

        # Average results
        rfsss[rep, 0] = 0.5 * (rfsss1[rep, 0] + rfsss2[rep, 0])
        rfsss[rep, 1] = np.sqrt(0.25 * (rfsss1[rep, 1] ** 2 + rfsss2[rep, 1] ** 2))

        rfgsss[rep, 0] = 0.5 * (rfgsss1[rep, 0] + rfgsss2[rep, 0])
        rfgsss[rep, 1] = np.sqrt(0.25 * (rfgsss1[rep, 1] ** 2 + rfgsss2[rep, 1] ** 2))

        rfsds[rep, 0] = 0.5 * (rfsds1[rep, 0] + rfsds2[rep, 0])
        rfsds[rep, 1] = np.sqrt(0.25 * (rfsds1[rep, 1] ** 2 + rfsds2[rep, 1] ** 2))

    # Create a dictionary with all the variables
    results = {"rfsss": rfsss, "rfgsss": rfgsss, "rfsds": rfsds}

    return results


def _create_toeplitz_cholesky(num_features, rho=0.7):
    """Create a Toeplitz matrix and return its Cholesky decomposition."""
    return la.cholesky(la.toeplitz([rho**i for i in range(num_features)]))


def _iid_se(x, e, xpxinv):
    """Calculate the iid standard error.

    Parameters:
    x : ndarray
        Input matrix (design matrix).
    e : ndarray
        Residual vector.
    xpxinv : ndarray
        Inverse of X'X.

    Returns:
    se : ndarray
        Standard errors.
    """
    k = x.shape[1] if len(x.shape) > 1 else 1
    n = x.shape[0]

    viid = (np.dot(e.T, e) / (n - k)) * np.atleast_2d(xpxinv)  # Ensure 2D matrix

    return np.sqrt(np.diag(viid)) if viid.ndim == 2 else np.sqrt(viid.item())


def _fail_if_invalid_inputs(
    n_replications, sample_size, num_features, alpha, rho, seed
):
    """Check if input parameters are valid."""
    if n_replications <= 0 or sample_size <= 0 or num_features <= 0:
        raise ValueError(
            "n_replications, sample_size, and num_features must be positive integers."
        )
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1.")
    if not (0 < rho < 1):
        raise ValueError("rho must be between 0 and 1.")
    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer.")
