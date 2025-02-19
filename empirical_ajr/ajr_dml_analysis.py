import numpy as np
import pandas as pd
import doubleml as dml
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

data = pd.read_csv('AJR_dataset.csv')

learner = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
ml_l = clone(learner)
ml_m = clone(learner)
ml_r = clone(learner)

np.random.seed(2222)

X = ['Latitude', 'Latitude2', 'Africa', 'Asia', 'Namer', 'Samer']

obj_dml_data = dml.DoubleMLData(data, 
                               y_col='GDP',
                               d_cols='Exprop',
                               z_cols='logMort',
                               x_cols=X)

# Run with n_folds=2
print("\nDouble ML PLIV Results with Random Forest (n_folds=2):")
print("----------------------------------")
dml_pliv_obj_2 = dml.DoubleMLPLIV(obj_dml_data, ml_l, ml_m, ml_r, n_folds=2, n_rep=100)
result_2 = dml_pliv_obj_2.fit()

print("\n------------------ Resampling        ------------------")
print(f"No. folds: {dml_pliv_obj_2.n_folds}")
print(f"No. repeated sample splits: {dml_pliv_obj_2.n_rep}")
print("\n------------------ Fit summary       ------------------")
print(result_2.summary)

# Run with n_folds=5
print("\nDouble ML PLIV Results with Random Forest (n_folds=5):")
print("----------------------------------")
dml_pliv_obj_5 = dml.DoubleMLPLIV(obj_dml_data, ml_l, ml_m, ml_r, n_folds=5, n_rep=100)
result_5 = dml_pliv_obj_5.fit()

print("\n------------------ Resampling        ------------------")
print(f"No. folds: {dml_pliv_obj_5.n_folds}")
print(f"No. repeated sample splits: {dml_pliv_obj_5.n_rep}")
print("\n------------------ Fit summary       ------------------")
print(result_5.summary)