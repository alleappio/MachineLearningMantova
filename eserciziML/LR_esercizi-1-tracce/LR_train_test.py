import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import LR_evaluation_complete as lre

# Load housing dataset
features_filename = 'datasets/features.dat'
targets_filename = 'datasets/targets.dat'
X = np.loadtxt(features_filename)
y = np.loadtxt(targets_filename)

X = np.hstack([np.ones((X.shape[0], 1)), X])

### prepeare training and testing datasets
#First, shuffle the dataset:
n= X.shape[0]
perm= np.random.permutation(n)
X= X[perm,...]
y= y[perm]
# Or, alternatively, use: X= np.random.shuffle(X)

#Second, split X in training and testing sets
split = 0.8
tr_size = int(n * split)
X_tr = X[0:tr_size,...]
y_tr = y[0:tr_size]
X_ts = X[tr_size:,...]
y_ts = y[tr_size:]


# Use sklearn to instantiate a Linear Regression algorithm
lr = LinearRegression()
# train regressor
lr.fit(X_tr, y_tr)
# test regressor
y_ts_hat = lr.predict(X_ts)

print('Evaluation using the sklearn metrics')
print("Error on train (MSE) = ", mean_squared_error(y_tr, lr.predict(X_tr)))
print("Error on test (MSE) = ", mean_squared_error(y_ts, y_ts_hat))
print("Error on train (MAE) = ", mean_absolute_error(y_tr, lr.predict(X_tr)))
print("Error on test (MAE) = ", mean_absolute_error(y_ts, y_ts_hat))

# TODO: Use your own metric functions on both the training and the testing datasets
print('Evaluation using my own metric implementations')


