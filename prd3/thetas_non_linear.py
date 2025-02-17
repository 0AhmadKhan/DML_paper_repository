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
# theta_hat - Non-Linear
# ==================================================================================

#region

def generate_theta_hat_non_linear(X, D, Y, g_X, m_X, U, V, trial):

    h_Y_X = conditional_expectation_Y_non_linear(X, D, Y, g_X, m_X, U, V, trial)
    # h_D_X = conditional_expectation_D_non_linear(X, D, Y, g_X, m_X, U, V, trial)

    theta_hat = np.mean((D)*(Y-h_Y_X)) / np.mean((D)**2)
    
    return theta_hat

#endregion


# ==================================================================================
# theta_check - Non-Linear
# ==================================================================================

#region

def generate_theta_check_non_linear(X, D, Y, g_X, m_X, U, V, trial):

    h_D_X = conditional_expectation_D_non_linear(X, D, Y, g_X, m_X, U, V, trial)
    h_Y_X = conditional_expectation_Y_non_linear(X, D, Y, g_X, m_X, U, V, trial)

    theta_check = np.mean((D-h_D_X)*(Y-h_Y_X)) / np.mean((D-h_D_X)**2)

    return theta_check

#endregion



