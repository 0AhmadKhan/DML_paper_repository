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
    
    U = np.random.normal(0, 1, n_samples) # noise as random normal as well
    V = np.random.normal(0, 1, n_samples)
    W = np.random.normal(0, 1, n_samples)
    
    D = m_X + V  # D as a scalar in the simple setting
    
    Y = D * theta + g_X + U # outcome variable

    #defining X4
    X4 = X1 + W
    X4_squared = X4 ** 2
    
    # stacking X1, X2, X3 into an array
    X = np.column_stack((X4, X2, X3, X4_squared, X2_squared, X3_squared))
    
    return X, D, Y, g_X, m_X, U, V



#endregion

