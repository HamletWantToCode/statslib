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

# data standardize
# class Standardizer(object):
#     def _mean(self, data):
#         mean_ = np.mean(data, axis=0)
#         self.mean_ = mean_
#         return mean_
    
#     def _var(self, data):
#         nsample = data.shape[0]
#         cov_mat = data.T @ data
#         var_ = np.sqrt(np.diag(cov_mat) / nsample)
#         self.var_ = var_
#         return var_
    
#     def fit_transform(self, data):
#         mean = self._mean(data)
#         var = self._var(data)
#         data -= mean
#         data /= var
#         return data
    
#     def transform(self, data):
#         data -= self.mean_
#         data /= self.var_
#         return data

# Tikhonov regularize
def regularizer(KM):
    def function(alpha):
        return alpha @ (KM @ alpha)
    return function

def regularizer_gradient(KM):
    def function(alpha):
        return 2*(KM @ alpha)
    return function

# math
def euclidean_distance(X, Y):
    XX = np.sum(X*X, axis=1)[:, np.newaxis]
    YY = np.sum(Y*Y, axis=1)[np.newaxis, :]
    distance = XX + YY - 2*(X @ Y.T)
    n = distance.shape[0]
    for i in range(n):
        distance[i, i] = 0      # the diagonal entry may small than zero due to numerical error
    return np.sqrt(distance)

def svd_solver(A, b, cond=None):
    U, S, Vh = np.linalg.svd(A)
    if cond is None:
        cond = np.finfo(np.float64).eps
    rank = len(S[S>cond])
    coefficients = np.divide((U.T[:rank] @ b), S[:rank])
    x = np.zeros(A.shape[1])
    for j, cj in enumerate(coefficients):
        x += cj*Vh[j]
    return x

# kernel
def check_array(X, Y):
    if Y is None:
        Y = X
    assert X.shape[1] == Y.shape[1], print('the 2nd dimension of X, Y not match !')
    return X, Y

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
    mse = np.sum((ypred - ytest)**2) / n
    return mse