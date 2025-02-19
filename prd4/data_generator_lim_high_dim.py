# ==================================================================================
# IMPORTS
# ==================================================================================
#region

from pathlib import Path
import numpy as np
import warnings

warnings.filterwarnings("ignore")

#endregion


# ==================================================================================
# Model Inputs - True Parameters
# ==================================================================================

#region

def true_parameters():
    params = {
        "theta" : 2.0, # true theta
        "beta" : [1.0, 0.5, -0.5, 0.2, -0.2, 0.1],  # true betas
        "eta" : [0.5, 0.5, 1.0, 0.3, -0.3, 0.4], # true etas
        "n_samples" : 500 # sample size 
        }
    return params

#endregion


# ==================================================================================
# Model Inputs - Linear
# ==================================================================================

#region

def generate_data_linear():
    """
    Generates data for the simulation of a partially linear model.
    
    Parameters:
        theta (float): The true treatment effect.
        beta (list of float): Coefficients for the nuisance function g(X).
        eta (list of float): Coefficients...
        n_samples (int): Number of samples to generate.
    
    Returns:
        X (ndarray): Generated covariates (shape: n_samples x 3).
        D (float): Scalar treatment variable.
        Y (ndarray): Generated outcome variable (shape: n_samples).
    """

    # Get parameters
    params = true_parameters()
    theta = params["theta"]
    beta = np.array(params["beta"])
    eta = np.array(params["eta"])
    n_samples = params["n_samples"]

    # X1, X2, X3 are random normal variables
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(0, 1, n_samples)
    X3 = np.random.normal(0, 1, n_samples)

    # Calculate squared terms
    X1_squared = X1 ** 2
    X2_squared = X2 ** 2
    X3_squared = X3 ** 2
    
    # nuisance function g(X) 
    g_X = X1 * beta[0] + X2 * beta[1] + X3 * beta[2] # + X1_squared * beta[3] + X2_squared * beta[4] + X3_squared * beta[5]
    
    # m(X)
    m_X = X1 * eta[0] + X2 * eta[1] + X3 * eta[2] # + X1_squared * eta[3] + X2_squared * eta[4] + X3_squared * eta[5]
    
    U = np.random.normal(0, 1, n_samples) # noise as random normal as well
    V = np.random.normal(0, 1, n_samples)

    
    D = m_X + V  # D as a scalar in the simple setting
    
    Y = D * theta + g_X + U # outcome variable
    
    # stacking X1, X2, X3 into an array
    X = np.column_stack((X1, X2, X3))
    
    return X, D, Y, g_X, m_X, U, V



#endregion


# ==================================================================================
# Model Inputs - Non-Linear in g(X) & m(X)
# ==================================================================================

#region

def generate_data_non_linear_gxmx():
    """
    Generates a 'harder' data scenario for partial linear / DML, with:
      - Correlated high-dimensional X
      - Highly nonlinear g(X), m(X)
      - Correlated errors U and V
    """

    # -------------------------------------------------------------------------
    # 1) True parameters and sample size
    # -------------------------------------------------------------------------
    params = true_parameters()
    theta = params["theta"]
    n_samples = params["n_samples"]  # e.g. 500

    # -------------------------------------------------------------------------
    # 2) Generate correlated covariates X (dimension = 10, say)
    #    We use a simple AR(1)-like correlation structure as an example.
    # -------------------------------------------------------------------------
    mean_vec = np.zeros(10)
    rho = 0.5  # correlation between neighboring X_i
    # Build covariance matrix with AR(1) pattern
    Sigma = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            Sigma[i, j] = rho ** abs(i - j)
    X_raw = np.random.multivariate_normal(mean=mean_vec, cov=Sigma, size=n_samples)

    # We'll call them X1,...,X10 for readability:
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X_raw.T

    # -------------------------------------------------------------------------
    # 3) Define complicated nuisance functions g(X) and m(X)
    #    (Here just a random mix of polynomials, exponentials, sines, etc.)
    # -------------------------------------------------------------------------
    # g(X): add various polynomial and trigonometric transformations
    g_X = (
        1.0 * np.sin(X1 * X2) +
        0.5 * (X3 ** 3) +
        np.exp(0.3 * X4) +
        0.25 * X5 * X6 +
        0.2 * (X7 ** 2) +
        np.cos(X8) +
        0.1 * X9 * X10 +
        0.05 * (X1 ** 2 + X2 ** 2 + X3 ** 2)  # just for extra complexity
    )

    # m(X): define the treatment equation also in a complicated way
    m_X = (
        0.8 * np.tanh(X1) +
        0.3 * X2 * X3 +
        0.2 * np.sin(X4 + X5) +
        0.1 * (X6 ** 2 + X7 ** 2) +
        0.05 * X8 * X9 * X10
    )

    # -------------------------------------------------------------------------
    # 4) Generate correlated errors U, V
    #    If you want to keep them independent, set correlation=0.
    #    But having them correlated can make the problem more challenging.
    # -------------------------------------------------------------------------
    corr_UV = 0.5  # correlation between U and V
    cov_mat = [[1, corr_UV],
               [corr_UV, 1]]
    uv = np.random.multivariate_normal(mean=[0, 0], cov=cov_mat, size=n_samples)
    U = uv[:, 0]
    V = uv[:, 1]

    # -------------------------------------------------------------------------
    # 5) Treatment and outcome
    # -------------------------------------------------------------------------
    D = m_X + V           # D = m(X) + V
    Y = theta * D + g_X + U  # Y = theta*D + g(X) + U

    # -------------------------------------------------------------------------
    # 6) Stack up X in a single array for the final return
    # -------------------------------------------------------------------------
    X = X_raw  # shape (n_samples, 10)

    return X, D, Y, g_X, m_X, U, V



#endregion

