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
    h = np.dot(X, theta)
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
    loss = np.dot((h - y).T, h - y) / 2 
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
        
        # Soluzione con cicli for: --------------------
        # for j in range(d):
        #     s= 0
        #     for i in range(n): 
        #         s= s + (y[i] - hyp(X[i], theta)) * X[i,j]
        #     theta[j] = theta[j] + alpha * s   
        # ---------------------------------------------
        
        # Soluzione con codice vettoriale: ------------
        J_grad = np.dot((hyp(X, theta) - y).T, X)
        theta -= alpha * J_grad
        # ---------------------------------------------
        
        
        Jold = Jnew
        Jnew = loss(X, y, theta)
    print("Optimization stopped, num iters = ", iter)
    return theta


# Soluzione con mini-batch di dimensione m 
# (il valore di m è dato in input)
# -----------------------------------------------------
# Linear regression solver - stochastic gradient descent
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
    
    print("\n Stochastic Gradient Descent ")

    max_iter = 10000
    n, d = X.shape
    theta = np.random.randn(d)

    Jold = np.inf
    perm= np.random.permutation(n)
    mini_batch_indexes= perm[:m]
    Mini_batch= X[mini_batch_indexes,...]
    Jnew = loss(Mini_batch, y[mini_batch_indexes], theta)

    iter = 0
    while np.abs(Jnew - Jold) >= eps and iter < max_iter:
        if np.mod(iter, 1000) == 0:
            print("iter ", iter, "-> J: ", Jnew)
        iter += 1
        
        perm= np.random.permutation(n)
        mini_batch_indexes= perm[:m]
        Mini_batch= X[mini_batch_indexes,...]
        
        J_grad = np.dot((hyp(Mini_batch, theta) - y[mini_batch_indexes]), Mini_batch)
        theta -= alpha * J_grad
        Jold = Jnew
        Jnew = loss(Mini_batch, y[mini_batch_indexes], theta)
    print("Optimization stopped, num iters = ", iter)
    return theta


# Soluzione con mini-batch di dimensione m = 1 (un solo sample)
# -----------------------------------------------------
# Linear regression solver - stochastic gradient descent
# -----------------------------------------------------
# def linear_regression_fit_SGD(X, y, alpha, eps=0.0001):
#     '''
#     :param y: target values
#     :param X: Feature matrix
#     :param alpha: learning rate
#     :param eps: stopping threshold
#     :param m: mini-batch size
#     :return: The updated regression weights
#     '''
    
#     print("\n Stochastic Gradient Descent ")

#     max_iter = 10000
#     n, d = X.shape
#     theta = np.random.randn(d)

#     Jold = np.inf
#     i= 0
#     Mini_batch= X[i,...]
#     Jnew = loss(Mini_batch, y[i], theta)

#     iter = 0
#     while np.abs(Jnew - Jold) >= eps and iter < max_iter:
#         if np.mod(iter, 1000) == 0:
#             print("iter ", iter, "-> J: ", Jnew)
#         iter += 1
        
#         Mini_batch= X[i,...]
#         err= hyp(Mini_batch, theta) - y[i]        
#         J_grad = err * Mini_batch
#         theta -= alpha * J_grad

#         Jold = Jnew
#         Jnew = loss(Mini_batch, y[i], theta)
#         i= i + 1
#         i= i % n
#     print("Optimization stopped, num iters = ", iter)
#    return theta




# -----------------------------------------------------
# Linear regression solver - Normal Equation
# -----------------------------------------------------
def linear_regression_fit_NE(X, y):
    '''
    :param y: target values
    :param X: Feature matrix
    :return: Linear regression weights
    '''
    theta_hat = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return theta_hat




'''
Compare NE, GD and SGD.
'''
# -----------------------------------------------------
# Hands On
# -----------------------------------------------------
if __name__ == "__main__":
    
    #House dataset:
    features_filename = 'datasets/features.dat'
    targets_filename = 'datasets/targets.dat'
    X = np.loadtxt(features_filename)
    y = np.loadtxt(targets_filename)
    mini_batch_size = 5 # qui fisso (arbitrariamente) la cardinalità del mini batch
    
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
    theta_SGD = linear_regression_fit_SGD(X2, y, alpha, mini_batch_size)
    stop_SGD = time()

    print("theta ottenuto tramite Normal Equation: ", theta_NE, " in ", (stop_NE - start_NE) * 1000, ' ms')
    print("theta ottenuto tramite Gradient Descent: ", theta_GD, " in ", (stop_GD - start_GD) * 1000, ' ms')
    print("theta ottenuto tramite Stochastic Gradient Descent: ", theta_SGD, " in ", (stop_SGD - start_SGD) * 1000, ' ms')



    ### Visualizzazione in 3D: -----------------------------------------------------
    
    # NE:

    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.grid()
    
    ax.scatter(X[:, 0], X[:, 1], y, c = 'r', s = 50)
    ax.set_title('Funzione ipotesi NE')
    
    px1 = np.arange(min(X[:, 0]), max(X[:, 0]), 0.2)
    px2 = np.arange(min(X[:, 1]), max(X[:, 1]), 0.2)
    PX1, PX2 = np.meshgrid(px1, px2)

    PY_NE= theta_NE[1] * PX1 + theta_NE[2] * PX2 + theta_NE[0]
    surf_NE = ax.plot_surface(PX1, PX2, PY_NE, cmap = plt.cm.cividis)
        
    ax.set_xlabel('$x_1$', labelpad=20)
    ax.set_ylabel('$x_2$', labelpad=20)
    ax.set_zlabel('y', labelpad=20)
    
    fig.colorbar(surf_NE, shrink=0.5, aspect=8)
    plt.show()
    

    # GD:

    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.grid()
    
    ax.scatter(X[:, 0], X[:, 1], y, c = 'r', s = 50)
    ax.set_title('Funzione ipotesi GD')
    
    PY_GD= theta_GD[1] * PX1 + theta_GD[2] * PX2 + theta_GD[0]
    surf_GD = ax.plot_surface(PX1, PX2, PY_GD, cmap = plt.cm.cividis)
        
    ax.set_xlabel('$x_1$', labelpad=20)
    ax.set_ylabel('$x_2$', labelpad=20)
    ax.set_zlabel('y', labelpad=20)

    fig.colorbar(surf_GD, shrink=0.5, aspect=8)
    plt.show()


    # SGD:

    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.grid()
    
    ax.scatter(X[:, 0], X[:, 1], y, c = 'r', s = 50)
    ax.set_title('Funzione ipotesi SGD')
    
    PY_SGD= theta_SGD[1] * PX1 + theta_SGD[2] * PX2 + theta_SGD[0]
    surf_SGD = ax.plot_surface(PX1, PX2, PY_SGD, cmap = plt.cm.cividis)
        
    ax.set_xlabel('$x_1$', labelpad=20)
    ax.set_ylabel('$x_2$', labelpad=20)
    ax.set_zlabel('y', labelpad=20)

    fig.colorbar(surf_GD, shrink=0.5, aspect=8)
    plt.show()


    ### Visualizzazione congiunta in 2D: -----------------------------------------------------

    ### Per ogni theta mostro la funzione ipotesi (proiettata sull'asse x1 del feature space)
    x1_pts= X[:, 0]
    y_pts_NE= theta_NE[1] * x1_pts + theta_NE[0]
    y_pts_GD= theta_GD[1] * x1_pts + theta_GD[0]
    y_pts_SGD= theta_SGD[1] * x1_pts + theta_SGD[0]

    plt.figure('Hypotheses')
    plt.title('Hypotheses ($x_1$ only)')
    plt.scatter(x1_pts, y)
    plt.plot(x1_pts, y_pts_NE, 'r', label='NE')
    plt.plot(x1_pts, y_pts_GD, 'g', label='GD')
    plt.plot(x1_pts, y_pts_SGD, 'b', label='SGD')
    plt.legend()
    plt.show()


    print("Valore finale loss function per NE: ", loss(X2, y, theta_NE))
    print("Valore finale loss function per GD: ", loss(X2, y, theta_GD))
    print("Valore finale loss function per SGD: ", loss(X2, y, theta_SGD))
    print("loss function NE <  loss function GD ?", loss(X2, y, theta_NE) < loss(X2, y, theta_GD))

    ### Inference -----------------------------------------------------
    # Dato un nuovo esempio x_test, calcolare (e mostrare) la predizione del modello corrispondente a theta_GD
    x_test= np.array([3500, 4]) # this is the input value
    
    x_test_augmented= np.array([1, x_test[0], x_test[1]])
    y_pred= hyp(x_test_augmented, theta_GD)
    print("y_pred: ", y_pred)
    
    
    plt.figure('Testing')
    plt.title('Prediction at inference time ($x_1$ only)')
    plt.scatter(x1_pts, y)
    plt.plot(x1_pts, y_pts_GD, 'g', label='GD')
    plt.scatter(x_test_augmented[1], y_pred, c='k')

    plt.legend()
    plt.show()    
    
 