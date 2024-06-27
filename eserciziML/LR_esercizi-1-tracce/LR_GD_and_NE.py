import numpy as np
from time import time
from matplotlib import pyplot as plt


# -----------------------------------------------------
# Hypothesis function
# -----------------------------------------------------
# hyp() prende in input il vettore dei patrametri theta e X
# X può essere sia un feature vector che una matrice di feature vector (e.g., la Design Matrix) 
def hyp(X, theta):
    '''
    :param X: Feature matrix
    :param theta: Linear regression weights
    :return: the value of the hypothesis function for each row of X
    '''
    h = 0 # TODO
    return h


# -----------------------------------------------------
# loss function
# -----------------------------------------------------
def loss(X, y, theta):
    '''
    :param y: target values
    :param X: Feature matrix
    :param theta: Linear regression weights
    :return: The loss function for the given input data
    '''
    h = hyp(X, theta)
    loss = 0 # TODO
    return loss


# -----------------------------------------------------
# Linear regression solver - gradient descent
# -----------------------------------------------------
def linear_regression_fit_GD(X, y, alpha, eps=0.0001):
    '''
    :param y: target values
    :param X: Feature matrix
    :param alpha: learning rate
    :param eps: stopping threshold
    :return: The updated regression weights
    '''

    print("\n Batch Gradient Descent ")
    max_iter = 10000
    n, d = X.shape
    theta = np.random.randn(d) # "standard" normal distribution
    Jold = np.inf
    Jnew = loss(X, y, theta)
    iter = 0
    while np.abs(Jnew - Jold) >= eps and iter < max_iter:
        if np.mod(iter, 1000) == 0:
            print("iter ", iter, "-> J: ", Jnew)
        iter += 1
        # TODO: calcolare il gradiente e aggiornare il vettore dei parametri
        #theta = None # TODO
        Jold = Jnew
        Jnew = loss(X, y, theta)
    print("Optimization stopped, num iters = ", iter)
    return theta


# -----------------------------------------------------
# Linear regression solver - gradient descent
# -----------------------------------------------------
def linear_regression_fit_SGD(X, y, alpha, m, eps=0.0001):
    '''
    :param y: target values
    :param X: Feature matrix
    :param alpha: learning rate
    :param eps: stopping threshold
    :param m: mini-batch size
    :return: The updated regression weights
    '''
    
    n, d = X.shape
    
    theta= np.random.randn(d) # TODO
    # Suggerimento: usare m = 1 (mini-batch da un solo elemento)
    
    return theta


# -----------------------------------------------------
# Linear regression solver - Normal Equation
# -----------------------------------------------------
def linear_regression_fit_NE(X, y):
    '''
    :param y: target values
    :param X: Feature matrix
    :return: Linear regression weights
    '''
    theta_hat = np.zeros((X.shape[1], 1)) # TODO
    # Suggerimento: usare il metodo numpy: linalg.inv()

    return theta_hat



'''
Compare NE, GD and SGD.
'''
# -----------------------------------------------------
# Hands On
# -----------------------------------------------------
if __name__ == "__main__":
    
    # House dataset:
    features_filename = 'datasets/features.dat'
    targets_filename = 'datasets/targets.dat'
    X = np.loadtxt(features_filename)
    y = np.loadtxt(targets_filename)
    
    ### EDA
    print(X.shape)
    print(y.shape)

    plt.style.use('seaborn-poster')
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.grid()
    ax.scatter(X[:, 0], X[:, 1], y, c = 'r', s = 50)
    ax.set_title('Training set')

    ax.set_xlabel('$x_1$', labelpad=20)
    ax.set_ylabel('$x_2$', labelpad=20)
    ax.set_zlabel('y', labelpad=20)

    plt.show()

    ### Aggiungo 1 al vettore delle feature:
    x_0 = np.ones((len(X), 1))
    X2 = np.append(x_0, X, axis=1)

    ### Calcolo 3 diversi parameter vector (theta) usando, rispettivamente: Gradient Descent, Stochastic Gradient Descent, Normal Equation
    start_NE = time()
    theta_NE = linear_regression_fit_NE(X2, y)
    stop_NE = time()
    
    alpha= 1e-9 # Fisso l'iperparametro del learning rate
    start_GD = time()
    theta_GD = linear_regression_fit_GD(X2, y, alpha)
    stop_GD = time()
    
    start_SGD = time()
    theta_SGD = linear_regression_fit_SGD(X2, y, alpha, 5)
    stop_SGD = time()

    print("theta ottenuto tramite Normal Equation: ", theta_NE, " in ", (stop_NE - start_NE) * 1000, ' ms')
    print("theta ottenuto tramite Gradient Descent: ", theta_GD, " in ", (stop_GD - start_GD) * 1000, ' ms')
    print("theta ottenuto tramite Stochastic Gradient Descent: ", theta_SGD, " in ", (stop_SGD - start_SGD) * 1000, ' ms')



    ### Inference -----------------------------------------------------
    # Dato un nuovo esempio x_test, calcolare (e mostrare) la predizione del modello corrispondente a theta_GD
    x_test= np.array([3500, 4]) # this is the input value
    
    # TODO

