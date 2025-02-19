import numpy as np
import pandas as pd
import doubleml as dml
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

learner = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
ml_l = clone(learner)
ml_m = clone(learner)

np.random.seed(1111)

data = pd.read_csv('penn_jae.csv', sep=r'\s+', skiprows=1)
data['inuidur1'] = np.log(data['inuidur1'])
data = data[(data['tg'] == 4) | (data['tg'] == 0)]

data['tg4'] = (data['tg'] == 4).astype(int)

X = ['female', 'black', 'othrace', 'dep', 'q2', 'q3', 'q4', 'q5', 'q6', 
          'agelt35', 'agegt54', 'durable', 'lusd', 'husd']

obj_dml_data = dml.DoubleMLData(data, y_col='inuidur1', d_cols='tg4', x_cols=X)

# Run with n_folds=2
print("\nDouble ML PLR Results with Random Forests (n_folds=2):")
print("----------------------------------")
dml_plr_obj_2 = dml.DoubleMLPLR(obj_dml_data, ml_l, ml_m, n_folds=2, n_rep=100)
result_2 = dml_plr_obj_2.fit()

print("\n------------------ Resampling        ------------------")
print(f"No. folds: {dml_plr_obj_2.n_folds}")
print(f"No. repeated sample splits: {dml_plr_obj_2.n_rep}")
print("\n------------------ Fit summary       ------------------")
print(result_2.summary)

# Run with n_folds=5
print("\nDouble ML PLR Results with Random Forests (n_folds=5):")
print("----------------------------------")
dml_plr_obj_5 = dml.DoubleMLPLR(obj_dml_data, ml_l, ml_m, n_folds=5, n_rep=100)
result_5 = dml_plr_obj_5.fit()

print("\n------------------ Resampling        ------------------")
print(f"No. folds: {dml_plr_obj_5.n_folds}")
print(f"No. repeated sample splits: {dml_plr_obj_5.n_rep}")
print("\n------------------ Fit summary       ------------------")
print(result_5.summary)