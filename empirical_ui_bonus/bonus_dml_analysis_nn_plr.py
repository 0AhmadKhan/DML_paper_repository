import numpy as np
import pandas as pd
import doubleml as dml
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import warnings

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Read the data
data = pd.read_csv('penn_jae.csv', sep=r'\s+', skiprows=1)
data['inuidur1'] = np.log(data['inuidur1'])
data = data[(data['tg'] == 4) | (data['tg'] == 0)]

# Create treatment indicator
data['tg4'] = (data['tg'] == 4).astype(int)

# Specify control variables (X)
x_cols = ['female', 'black', 'othrace', 'dep', 'q2', 'q3', 'q4', 'q5', 'q6', 
          'agelt35', 'agegt54', 'durable', 'lusd', 'husd']

# Set up the neural network learner
learner = MLPRegressor(
    hidden_layer_sizes=(32, 16),  # Two hidden layers
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=2222,
    early_stopping=True
)

# Clone the learner for each model
ml_l = clone(learner)
ml_m = clone(learner)

# Set random seed for reproducibility
np.random.seed(2222)

# Create DoubleML data object with polynomial features
obj_dml_data = dml.DoubleMLData(data, 
                               y_col='inuidur1',
                               d_cols='tg4',
                               x_cols=x_cols)

# Run with n_folds=2
print("\nDouble ML PLR Results with Neural Networks (n_folds=2):")
print("----------------------------------")
dml_plr_obj_2 = dml.DoubleMLPLR(obj_dml_data, ml_l, ml_m, n_folds=2, n_rep=100)
result_2 = dml_plr_obj_2.fit()

# Print only resampling and coefficient table for n_folds=2
print("\n------------------ Resampling        ------------------")
print(f"No. folds: {dml_plr_obj_2.n_folds}")
print(f"No. repeated sample splits: {dml_plr_obj_2.n_rep}")
print("\n------------------ Fit summary       ------------------")
print(result_2.summary)

# Run with n_folds=5
print("\nDouble ML PLR Results with Neural Networks (n_folds=5):")
print("----------------------------------")
dml_plr_obj_5 = dml.DoubleMLPLR(obj_dml_data, ml_l, ml_m, n_folds=5, n_rep=100)
result_5 = dml_plr_obj_5.fit()

# Print only resampling and coefficient table for n_folds=5
print("\n------------------ Resampling        ------------------")
print(f"No. folds: {dml_plr_obj_5.n_folds}")
print(f"No. repeated sample splits: {dml_plr_obj_5.n_rep}")
print("\n------------------ Fit summary       ------------------")
print(result_5.summary)