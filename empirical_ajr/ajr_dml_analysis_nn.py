import numpy as np
import pandas as pd
import doubleml as dml
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

data = pd.read_csv('AJR_dataset.csv')


X = ['Latitude', 'Latitude2', 'Africa', 'Asia', 'Namer', 'Samer']

# Set up the neural network learner
learner = MLPRegressor(
    hidden_layer_sizes=(32, 16),  # Two hidden layers
    activation='relu',
    solver='adam',
    max_iter=5000,
    random_state=2222,
    early_stopping=True
)

# Clone the learner for each model
ml_l = clone(learner)
ml_m = clone(learner)
ml_r = clone(learner)

# Set random seed for reproducibility
np.random.seed(2222)

obj_dml_data = dml.DoubleMLData(data, 
                               y_col='GDP',
                               d_cols='Exprop',
                               z_cols='logMort',
                               x_cols=X)



# Run with n_folds=5
print("\nDouble ML PLIV Results with Neural Networks (n_folds=2):")
print("----------------------------------")
dml_pliv_obj_2 = dml.DoubleMLPLIV(obj_dml_data, ml_l, ml_m, ml_r, n_folds=2, n_rep=100)
result_2 = dml_pliv_obj_2.fit()

# Print only resampling and coefficient table for n_folds=2
print("\n------------------ Resampling        ------------------")
print(f"No. folds: {dml_pliv_obj_2.n_folds}")
print(f"No. repeated sample splits: {dml_pliv_obj_2.n_rep}")
print("\n------------------ Fit summary       ------------------")
print(result_2.summary)

# Run with n_folds=2
print("\nDouble ML PLIV Results with Neural Networks (n_folds=5):")
print("----------------------------------")
dml_pliv_obj_5 = dml.DoubleMLPLIV(obj_dml_data, ml_l, ml_m, ml_r, n_folds=5, n_rep=100)
result_5 = dml_pliv_obj_5.fit()

# Print only resampling and coefficient table for n_folds=5
print("\n------------------ Resampling        ------------------")
print(f"No. folds: {dml_pliv_obj_5.n_folds}")
print(f"No. repeated sample splits: {dml_pliv_obj_5.n_rep}")
print("\n------------------ Fit summary       ------------------")
print(result_5.summary)