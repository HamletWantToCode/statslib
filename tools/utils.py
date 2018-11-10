# miscellanious utility

import numpy as np 

# dataset
def load_data(data, n_batch=1):
    N = data.shape[0]
    n_elements = N//n_batch
    for i in range(n_batch):
        partition = data[i*n_elements : (i+1)*n_elements]
        feature, target = partition[:, :-1], partition[:, -1]
        yield (feature, target)

# Tikhonov regularize
def regularizer(KM):
    def function(alpha):
        return alpha @ (KM @ alpha)
    return function

def regularizer_gradient(KM):
    def function(alpha):
        return KM @ alpha
    return function

# kernel
def check_array(X, Y):
    if Y is None:
        Y = X
    assert X.shape[1] == Y.shape[1], print('the 2nd dimension of X, Y not match !')
    return X, Y

def euclidean_distance(X, Y):
    XX = np.sum(X*X, axis=1)[:, np.newaxis]
    YY = np.sum(Y*Y, axis=1)[np.newaxis, :]
    distance = XX + YY - 2*(X @ Y.T)
    return np.sqrt(distance)

def linearKernel(X, Y=None):
    X, Y = check_array(X, Y)
    return X @ Y.T

def rbfKernel(gamma):
    def rbf_function(X, Y=None):
        X, Y = check_array(X, Y)
        square_distance = (euclidean_distance(X, Y))**2
        return np.exp(-gamma*square_distance)
    return rbf_function

def laplaceKernel(gamma):
    def ll_function(X, Y=None):
        X, Y = check_array(X, Y)
        distance = euclidean_distance(X, Y)    
        return np.exp(-gamma*distance)
    return ll_function

def polyKernel(gamma, r0, d):
    def poly_function(X, Y=None):
        X, Y = check_array(X, Y)
        return (r0 + gamma*(X @ Y.T))**d
    return poly_function

# metric
def classifyAccuracy(ypred, ytest):
    n = len(ytest)
    assert len(ypred) == n
    accuracy = 0
    for i, label in enumerate(ypred):
        if label == ytest[i]:
            accuracy += 1
    return accuracy*1.0 / n

def meanSquareError(ypred, ytest):
    n = len(ytest)
    assert len(ypred) == n
    mse = np.sqrt(np.sum((ypred - ytest)**2)) / n
    return mse