# miscellanious utility

import numpy as np

# dataset
def load_data(X, y, n_batch=1):
    N = X.shape[0]
    n_elements = N//n_batch
    for i in range(n_batch):
        feature, target = X[i*n_elements : (i+1)*n_elements], y[i*n_elements : (i+1)*n_elements]
        yield (feature, target)

# data standardize
# class Standardizer(object):
#     def _mean(self, data):
#         mean_ = np.mean(data, axis=0)
#         self.mean_ = mean_
#         return mean_

#     def _var(self, center_data):
#         """
#         NOTE: this 'variance' is different from the
#         variance in math book, I don't divide the sample
#         number. I want to make the singular value of
#         transformed data to be 1, this will make optimization
#         process easier
#         """
#         nsample = center_data.shape[0]
#         cov_mat = center_data.T @ center_data
#         var_ = np.sqrt(np.diag(cov_mat) / nsample)
#         self.var_ = var_
#         return var_

#     def fit_transform(self, data):
#         mean = self._mean(data)
#         center_data = data - mean
#         var = self._var(center_data)
#         normal_data = center_data / var
#         return normal_data

#     def transform(self, data):
#         center_data = data - self.mean_
#         normal_data = data / self.var_
#         return normal_data

# PCA transformation (for real matrix)
class PCA_transform(object):
    def __init__(self, n_components):
        self.ncmp_ = n_components

    def fit_transform(self, data):
        cov_mat = data.T @ data
        U, S, _ = np.linalg.svd(cov_mat)
        self.trans_mat = U[:, :self.ncmp_]
        return data @ U[:, :self.ncmp_]

    def transform(self, data):
        return data @ self.trans_mat

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
    # special care about complex entry
    X_ = X[:, np.newaxis, :]
    Y_ = Y[np.newaxis, :, :]
    D_ = X_ - Y_
    distance = np.sum(D_*D_.conj(), axis=2, dtype=np.float64)
    return np.sqrt(distance)

def manhattan_distance(X, Y):
    # special care about complex entry
    X_ = X[:, np.newaxis, :]
    Y_ = Y[np.newaxis, :, :]
    D_ = X_ - Y_
    M = np.sum(np.sqrt(D_*D_.conj()), axis=2, dtype=np.float64)
    return M

## used for real symmetric matrix
def svd_solver(A, b, cond=None):
    U, S, _ = np.linalg.svd(A)
    if cond is None:
        cond = np.finfo(np.float64).eps
    rank = len(S[S>cond])
    coefficients = np.squeeze(U.T[:rank] @ b) / S[:rank]
    x = np.sum(coefficients[np.newaxis, :] * U[:, :rank], axis=1)
    return x

# kernel
def check_array(X, Y):
    if Y is None:
        Y = X
    assert X.shape[1] == Y.shape[1], print('the 2nd dimension of X, Y not match !')
    return X, Y

def linearKernel(X, Y=None):
    X, Y = check_array(X, Y)
    return X @ (Y.conj()).T

def rbfKernel(gamma):
    def rbf_function(X, Y=None):
        X, Y = check_array(X, Y)
        square_distance = (euclidean_distance(X, Y))**2
        return np.exp(-gamma*square_distance)
    return rbf_function

def rbfKernel_gd(gamma):
    def rbf_gd(X, Y=None):
        X, Y = check_array(X, Y)
        square_distance = (euclidean_distance(X, Y))**2
        K = np.exp(-gamma*square_distance)
        diff = X[:, np.newaxis, :] - Y
        gd = -2*gamma*diff*K[:, :, np.newaxis]
        return np.transpose(gd, axes=(0, 2, 1))
    return rbf_gd

def laplaceKernel(gamma):
    def ll_function(X, Y=None):
        X, Y = check_array(X, Y)
        distance = manhattan_distance(X, Y)
        return np.exp(-gamma*distance)
    return ll_function

def laplaceKernel_gd(gamma):
    def ll_gd(X, Y=None):
        X, Y = check_array(X, Y)
        distance = manhattan_distance(X, Y)
        K = np.exp(-gamma*distance)
        diff = X[:, np.newaxis, :] - Y
        mod_diff = np.sqrt(diff*diff.conj()).real
        np.maximum(1e-15, mod_diff, out=mod_diff)
        gd = -gamma*(diff/mod_diff)*K[:, :, np.newaxis]
        return np.transpose(gd, axes=(0, 2, 1))
    return ll_gd

def polyKernel(gamma, r0, d):
    def poly_function(X, Y=None):
        X, Y = check_array(X, Y)
        return (r0 + gamma*(X @ (Y.conj()).T))**d
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
