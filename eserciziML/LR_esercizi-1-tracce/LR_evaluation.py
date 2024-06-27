import numpy as np
from time import time
from matplotlib import pyplot as plt



# -----------------------------------------------------
# Residual Sum of Squares
# -----------------------------------------------------
def RSS(y_true, y_pred):
    '''
    :param y_true: target values
    :param y_pred: predicted y
    :return: RSS
    '''
    # TODO


# -----------------------------------------------------
# Residual Standard Error
# -----------------------------------------------------
def RSE(y_true, y_pred, X):
    '''
    :param y_true: target values
    :param y_pred: predicted y
    :param X: Design matrix
    :return: RSE
    '''
    # TODO


# -----------------------------------------------------
# Mean Squared Error
# -----------------------------------------------------
def MSE(y_true, y_pred):
    '''
    :param y_true: target values
    :param y_pred: predicted y
    :return: MSE
    '''
    n=len(y_true)
    return RSS(y_true,y_pred)/n


# -----------------------------------------------------
# Mean Absolute Error
# -----------------------------------------------------
def MAE(y_true, y_pred):
    '''
    :param y_true: target values
    :param y_pred: predicted y
    :return: MAE
    '''
    # TODO


# -----------------------------------------------------
# Root Mean Square Error
# -----------------------------------------------------
def RMSE(y_true, y_pred):
    '''
    :param y_true: target values
    :param y_pred: predicted y
    :return: RMSE
    '''
    # TODO
