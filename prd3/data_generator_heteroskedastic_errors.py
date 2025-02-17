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
        "n_samples" : 500 # sample size 
        }
    return params

#endregion



# ==================================================================================
# Model Inputs - Non-Linear in g(X) & m(X)
# ==================================================================================

#region

def generate_data_non_linear_gxmx():
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
    g_X = (X1 * beta[0] + X2 * beta[1] + X3 * beta[2]
           + X1_squared * beta[3] + X2_squared * beta[4] + X3_squared * beta[5])
    
    # m(X)
    m_X = (X1 * eta[0] + X2 * eta[1] + X3 * eta[2]
           + X1_squared * eta[3] + X2_squared * eta[4] + X3_squared * eta[5])
    
    
    sigma_U = 1 + 0.5 * np.abs(X1 + X2)
    sigma_V = 1 + 0.5 * np.abs(X2 + X3)

    U = np.random.normal(0, sigma_U, n_samples) # noise as random normal as well
    V = np.random.normal(0, sigma_V, n_samples)
        
    D = m_X + V  # D as a scalar in the simple setting
    
    Y = D * theta + g_X + U # outcome variable
    
    
    # stacking X1, X2, X3 into an array
    X = np.column_stack((X1, X2, X3, X1_squared, X2_squared, X3_squared))
    
    return X, D, Y, g_X, m_X, U, V



#endregion

