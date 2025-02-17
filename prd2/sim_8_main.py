# ==================================================================================
# IMPORTS
# ==================================================================================
#region

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

from sim_8_functions import *
from sim_8_thetas_linear import *
from sim_8_thetas_non_linear import *
from sim_8_plot import *

face_colors = sns.color_palette('pastel')
edge_colors = sns.color_palette('dark')

warnings.filterwarnings("ignore")

#endregion


if __name__ == "__main__":
    
    n_trials = 100

    # ==================================================================================
    # Linear thetas
    # ==================================================================================
    '''
    #region

    theta_hat_estimates_linear   = []
    theta_check_estimates_linear = []
    theta_tilde_estimates_linear = []
    theta_breve_estimates_linear = []
    

    for trial in range(n_trials):

        # generating data
        X, D, Y, g_X, m_X, U, V = generate_data_linear()

        theta_hat = generate_theta_hat_linear(X, D, Y, g_X, m_X, U, V)
        theta_hat_estimates_linear.append(theta_hat)

        theta_check = generate_theta_check_linear(X, D, Y, g_X, m_X, U, V)
        theta_check_estimates_linear.append(theta_check)

        theta_tilde = generate_theta_tilde_linear(X, D, Y, g_X, m_X, U, V)
        theta_tilde_estimates_linear.append(theta_tilde)

        theta_breve = generate_theta_breve_linear(X, D, Y, g_X, m_X, U, V)
        theta_breve_estimates_linear.append(theta_breve)

    theta_hat_array   = np.array(theta_hat_estimates_linear  )
    theta_check_array = np.array(theta_check_estimates_linear)
    theta_tilde_array = np.array(theta_tilde_estimates_linear)
    theta_breve_array = np.array(theta_breve_estimates_linear)

    data = {
    "Theta_Hat": theta_hat_array,
    "Theta_Check": theta_check_array,
    "Theta_Tilde": theta_tilde_array,
    "Theta_Breve": theta_breve_array
    }

    raw_data = pd.DataFrame(data)

    fig = generate_plots(raw_data)

    # ==================================================================================
    # Save linear plot
    # ==================================================================================
    BLD = Path("bld2")
    if not BLD.exists():
        BLD.mkdir()
    # data.to_pickle(BLD / "results.pkl")
    # fig.write_image(BLD / "bias.png")
    fig.savefig(BLD / "linear_scaled.png")
    plt.close(fig)


    #endregion
    '''

    # ==================================================================================
    # Non-Linear in g(X) thetas
    # ==================================================================================
    '''
    #region

    theta_hat_estimates_non_linear_gx   = []
    theta_tilde_estimates_non_linear_gx = []
    theta_check_estimates_non_linear_gx = []
    theta_breve_estimates_non_linear_gx = []
    

    for trial in range(n_trials):

        # generating data
        X, D, Y, g_X, m_X, U, V = generate_data_non_linear_gx()

        theta_hat = generate_theta_hat_non_linear(X, D, Y, g_X, m_X, U, V, trial)
        theta_hat_estimates_non_linear_gx.append(theta_hat)

        theta_check = generate_theta_check_non_linear(X, D, Y, g_X, m_X, U, V, trial)
        theta_check_estimates_non_linear_gx.append(theta_check)

        theta_tilde = generate_theta_tilde_non_linear(X, D, Y, g_X, m_X, U, V, trial)
        theta_tilde_estimates_non_linear_gx.append(theta_tilde)

        theta_breve = generate_theta_breve_non_linear(X, D, Y, g_X, m_X, U, V, trial)
        theta_breve_estimates_non_linear_gx.append(theta_breve)


    theta_hat_array   = np.array(theta_hat_estimates_non_linear_gx  )
    theta_check_array = np.array(theta_tilde_estimates_non_linear_gx)
    theta_tilde_array = np.array(theta_check_estimates_non_linear_gx)
    theta_breve_array = np.array(theta_breve_estimates_non_linear_gx)

    data = {
    "Theta_Hat": theta_hat_array,
    "Theta_Check": theta_check_array,
    "Theta_Tilde": theta_tilde_array,
    "Theta_Breve": theta_breve_array
    }

    raw_data = pd.DataFrame(data)

    fig = generate_plots(raw_data)


    # ==================================================================================
    # Save non-linear in g(X) plot
    # ==================================================================================
    BLD = Path("bld2")
    if not BLD.exists():
        BLD.mkdir()
    # data.to_pickle(BLD / "results.pkl")
    # fig.write_image(BLD / "bias.png")
    plt.savefig(BLD / "non_linear_gx_scaled.png")
    plt.close()

    #endregion
    '''


    # ==================================================================================
    # Non-Linear in g(X) and m(X) thetas
    # ==================================================================================
    '''
    #region

    theta_hat_estimates_non_linear_gxmx   = []
    theta_tilde_estimates_non_linear_gxmx = []
    theta_check_estimates_non_linear_gxmx = []
    theta_breve_estimates_non_linear_gxmx = []
    

    for trial in range(n_trials):

        # generating data
        X, D, Y, g_X, m_X, U, V = generate_data_non_linear_gxmx()

        theta_hat = generate_theta_hat_non_linear(X, D, Y, g_X, m_X, U, V, trial)
        theta_hat_estimates_non_linear_gxmx.append(theta_hat)

        theta_check = generate_theta_check_non_linear(X, D, Y, g_X, m_X, U, V, trial)
        theta_check_estimates_non_linear_gxmx.append(theta_check)

        theta_tilde = generate_theta_tilde_non_linear(X, D, Y, g_X, m_X, U, V, trial)
        theta_tilde_estimates_non_linear_gxmx.append(theta_tilde)

        theta_breve = generate_theta_breve_non_linear(X, D, Y, g_X, m_X, U, V, trial)
        theta_breve_estimates_non_linear_gxmx.append(theta_breve)

    theta_hat_array   = np.array(theta_hat_estimates_non_linear_gxmx  )
    theta_check_array = np.array(theta_tilde_estimates_non_linear_gxmx)
    theta_tilde_array = np.array(theta_check_estimates_non_linear_gxmx)
    theta_breve_array = np.array(theta_breve_estimates_non_linear_gxmx)

    data = {
    "Theta_Hat": theta_hat_array,
    "Theta_Check": theta_check_array,
    "Theta_Tilde": theta_tilde_array,
    "Theta_Breve": theta_breve_array
    }

    raw_data = pd.DataFrame(data)

    fig = generate_plots(raw_data)

    # ==================================================================================
    # Save linear plot
    # ==================================================================================
    BLD = Path("bld2")
    if not BLD.exists():
        BLD.mkdir()
    # data.to_pickle(BLD / "results.pkl")
    # fig.write_image(BLD / "bias.png")
    plt.savefig(BLD / "non_linear_gxmx.png")
    plt.close()

    #endregion
    '''


    # ==================================================================================
    # Linear thetas with rf
    # ==================================================================================
    '''
    #region

    theta_hat_estimates_linear   = []
    theta_check_estimates_linear = []
    theta_tilde_estimates_linear = []
    theta_breve_estimates_linear = []
    

    for trial in range(n_trials):

        # generating data
        X, D, Y, g_X, m_X, U, V = generate_data_linear()

        theta_hat = generate_theta_hat_non_linear(X, D, Y, g_X, m_X, U, V, trial)
        theta_hat_estimates_linear.append(theta_hat)

        theta_check = generate_theta_check_non_linear(X, D, Y, g_X, m_X, U, V, trial)
        theta_check_estimates_linear.append(theta_check)

        theta_tilde = generate_theta_tilde_non_linear(X, D, Y, g_X, m_X, U, V, trial)
        theta_tilde_estimates_linear.append(theta_tilde)

        theta_breve = generate_theta_breve_non_linear(X, D, Y, g_X, m_X, U, V, trial)
        theta_breve_estimates_linear.append(theta_breve)

    theta_hat_array   = np.array(theta_hat_estimates_linear  )
    theta_check_array = np.array(theta_check_estimates_linear)
    theta_tilde_array = np.array(theta_tilde_estimates_linear)
    theta_breve_array = np.array(theta_breve_estimates_linear)

    data = {
    "Theta_Hat": theta_hat_array,
    "Theta_Check": theta_check_array,
    "Theta_Tilde": theta_tilde_array,
    "Theta_Breve": theta_breve_array
    }

    raw_data = pd.DataFrame(data)

    fig = generate_plots(raw_data)


    # ==================================================================================
    # Save linear plot
    # ==================================================================================
    BLD = Path("bld2")
    if not BLD.exists():
        BLD.mkdir()
    # data.to_pickle(BLD / "results.pkl")
    # fig.write_image(BLD / "bias.png")
    plt.savefig(BLD / "linear_with_rf_scaled.png")
    plt.close()

    #endregion
    '''
































