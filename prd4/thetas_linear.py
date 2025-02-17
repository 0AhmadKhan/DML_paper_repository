# ==================================================================================
# IMPORTS
# ==================================================================================
#region

from pathlib import Path
import numpy as np
import warnings

from functions import *
# from sim_8_functions import (
#     conditional_expectation_Y_linear, 
#     conditional_expectation_D_linear,
#     conditional_expectation_g_hat_X_linear
# )

warnings.filterwarnings("ignore")

#endregion



# ==================================================================================
# theta_hat - Linear
# ==================================================================================

#region

def generate_theta_hat_linear(X, D, Y, g_X, m_X, U, V):

    h_Y_X = conditional_expectation_Y_linear(X, D, Y, g_X, m_X, U, V)
    h_D_X = conditional_expectation_D_linear(X, D, Y, g_X, m_X, U, V)

    theta_hat = np.mean((D-h_D_X)*(Y)) / np.mean((D-h_D_X)**2)
    
    return theta_hat

#endregion


# ==================================================================================
# theta_check - Linear
# ==================================================================================

#region

def generate_theta_check_linear(X, D, Y, g_X, m_X, U, V):

    h_D_X = conditional_expectation_D_linear(X, D, Y, g_X, m_X, U, V)
    h_Y_X = conditional_expectation_Y_linear(X, D, Y, g_X, m_X, U, V)

    theta_check = np.mean((D-h_D_X)*(Y-h_Y_X)) / np.mean((D-h_D_X)**2)

    return theta_check

#endregion

























