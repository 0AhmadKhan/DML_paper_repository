import numpy as np
import pandas as pd
import doubleml as dml
from sklearn.base import clone
import warnings
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

warnings.filterwarnings('ignore', category=FutureWarning)

ml_g = RandomForestRegressor(n_estimators=100, max_features=10, max_depth=5, min_samples_leaf=2)

ml_m = RandomForestClassifier(n_estimators=100, max_features=10, max_depth=5, min_samples_leaf=2)

np.random.seed(3333)

data = pd.read_csv('penn_jae.csv', sep=r'\s+', skiprows=1)
data = data[(data['tg'] == 4) | (data['tg'] == 0)]
data['inuidur1'] = np.log(data['inuidur1'])
data['tg4'] = (data['tg'] == 4).astype(int)

X = ['female', 'black', 'othrace', 'dep', 'q2', 'q3', 'q4', 'q5', 'q6', 
          'agelt35', 'agegt54', 'durable', 'lusd', 'husd']

obj_dml_data = dml.DoubleMLData(data, y_col='inuidur1', d_cols='tg4', x_cols=X)


# Run with n_folds=2
print("\nDouble ML IRM Results with Random Forests (n_folds=2):")
print("----------------------------------")
dml_irm_obj_2 = dml.DoubleMLIRM(obj_dml_data, ml_g, ml_m, n_folds=2, n_rep=100)
result_2 = dml_irm_obj_2.fit()

print("\n------------------ Resampling        ------------------")
print(f"No. folds: {dml_irm_obj_2.n_folds}")
print(f"No. repeated sample splits: {dml_irm_obj_2.n_rep}")
print("\n------------------ Fit summary       ------------------")
print(result_2.summary)

# Run with n_folds=5
print("\nDouble ML IRM Results with Random Forests (n_folds=5):")
print("----------------------------------")
dml_irm_obj_5 = dml.DoubleMLIRM(obj_dml_data, ml_g, ml_m, n_folds=5, n_rep=100)
result_5 = dml_irm_obj_5.fit()

print("\n------------------ Resampling        ------------------")
print(f"No. folds: {dml_irm_obj_5.n_folds}")
print(f"No. repeated sample splits: {dml_irm_obj_5.n_rep}")
print("\n------------------ Fit summary       ------------------")
print(result_5.summary)




