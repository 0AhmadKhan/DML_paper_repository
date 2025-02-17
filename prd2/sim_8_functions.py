# ==================================================================================
# IMPORTS
# ==================================================================================
#region

from pathlib import Path
import warnings
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

#endregion


# ==================================================================================
# h_Y(X) - Linear
# ==================================================================================

#region

def conditional_expectation_Y_linear(X, D, Y, g_X, m_X, U, V):

    # Fit linear regression model (to get h_Y(X))
    ols_model = LinearRegression(fit_intercept=False)
    ols_model.fit(X, Y)
    h_Y_X = ols_model.predict(X)

    return h_Y_X

#endregion


# ==================================================================================
# h_D(X) - Linear
# ==================================================================================

#region

def conditional_expectation_D_linear(X, D, Y, g_X, m_X, U, V):

    # D_for_OLS = D.reshape(-1,1) # Doesn't seem necessary when using predict 

    # Fit linear regression model (to get h_D(X))
    ols_model = LinearRegression(fit_intercept=False)
    ols_model.fit(X, D)
    h_D_X = ols_model.predict(X)

    return h_D_X

#endregion


# ==================================================================================
# h_Y(X) - Non-Linear
# ==================================================================================

#region

def conditional_expectation_Y_non_linear(X, D, Y, g_X, m_X, U, V, trial):

    # Fit Random Forest (to get h_Y(X))
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=trial)
    rf_model.fit(X, Y)  
    h_Y_X = rf_model.predict(X)  
     
    return h_Y_X

#endregion

# ==================================================================================
# h_D(X) - Non-Linear
# ==================================================================================

#region

def conditional_expectation_D_non_linear(X, D, Y, g_X, m_X, U, V, trial):

    # Fit Random Forest (to get h_D(X))
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=trial)
    rf_model.fit(X, D)  
    h_D_X = rf_model.predict(X)  
     
    return h_D_X

#endregion


