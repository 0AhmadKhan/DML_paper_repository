# ==================================================================================
# IMPORTS
# ==================================================================================
#region

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import warnings
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.base import clone

from doubleml import DoubleMLData
from doubleml import DoubleMLPLR
from doubleml.datasets import make_plr_CCDDHNR2018

face_colors = sns.color_palette('pastel')
edge_colors = sns.color_palette('dark')

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
        "n_samples" : 20 # sample size 
        }
    return params

#endregion



# ==================================================================================
# Model Inputs - Non-Linear in g(X) & m(X)
# ==================================================================================

#region

def generate_data_non_linear_gxmx():
    """
    Generates data for the simulation of a partially linear model with high-dimensional covariates
    and sparsity in g(X) and m(X).

    Returns:
        X (ndarray): Generated covariates (shape: n_samples x p).
        D (ndarray): Treatment variable (shape: n_samples).
        Y (ndarray): Outcome variable (shape: n_samples).
        g_X (ndarray): Nuisance function g(X) values (shape: n_samples).
        m_X (ndarray): Nuisance function m(X) values (shape: n_samples).
        U (ndarray): Noise term for the outcome (shape: n_samples).
        V (ndarray): Noise term for the treatment (shape: n_samples).
    """

    # Get parameters
    params = true_parameters()
    theta = params["theta"]
    # beta = np.array(params["beta"])
    # eta = np.array(params["eta"])
    n_samples = params["n_samples"]
    p = n_samples*2

    # # X1, X2, X3 are random normal variables
    # X1 = np.random.normal(0, 1, n_samples)
    # X2 = np.random.normal(0, 1, n_samples)
    # X3 = np.random.normal(0, 1, n_samples)

    # # Calculate squared terms
    # X1_squared = X1 ** 2
    # X2_squared = X2 ** 2
    # X3_squared = X3 ** 2

    # # nuisance function g(X) 
    # g_X = (X1 * beta[0] + X2 * beta[1] + X3 * beta[2]
    #        + X1_squared * beta[3] + X2_squared * beta[4] + X3_squared * beta[5])
    
    # # m(X)
    # m_X = (X1 * eta[0] + X2 * eta[1] + X3 * eta[2]
    #        + X1_squared * eta[3] + X2_squared * eta[4] + X3_squared * eta[5])

    
    # Generate high-dimensional covariates (X1, X2, ..., Xp)
    X = np.random.normal(0, 1, (n_samples, p))

    # Sparsity: Randomly pick sparse indices for g(X) and m(X)
    sparsity_g = np.random.choice(p, size=10, replace=False)  # 10 non-zero coefficients in g(X)
    sparsity_m = np.random.choice(p, size=10, replace=False)  # 10 non-zero coefficients in m(X)

    # Generate coefficients for g(X) and m(X) with sparsity
    beta = np.zeros(p)
    eta = np.zeros(p)
        
    # Assign random coefficients to sparse indices
    beta[sparsity_g] = np.random.uniform(-1, 1, len(sparsity_g))
    eta[sparsity_m] = np.random.uniform(-1, 1, len(sparsity_m))

    # Calculate g(X) using sparse beta
    g_X = X @ beta

    # Calculate m(X) using sparse eta
    m_X = X @ eta
    
    U = np.random.normal(0, 1, n_samples) # noise as random normal as well
    V = np.random.normal(0, 1, n_samples)
    W = np.random.normal(0, 1, n_samples)
    
    D = m_X + V  # D as a scalar in the simple setting
    
    Y = D * theta + g_X + U # outcome variable
    
      
    return X, D, Y, g_X, m_X, U, V



#endregion

