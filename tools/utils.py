# miscellanious utility

import numpy as np

# dataset
def load_data(X, y, n_batch=1):
    N = X.shape[0]
    n_elements = N//n_batch
    for i in range(n_batch):
        feature, target = X[i*n_elements : (i+1)*n_elements], y[i*n_elements : (i+1)*n_elements]
        yield (feature, target)

# data-split
def n_split(nsplits, Num, random_state):
    np.random.seed(random_state)
    index = np.arange(0, Num, 1, dtype=int)
    np.random.shuffle(index)
    for i in range(nsplits):
        part1 = index[i:Num:nsplits]
        part2 = np.setdiff1d(index, part1)
        yield part1, part2

def stratified_split(nsplits, Num, labels, random_state):
    pass

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
        (N1, D), N = X.shape, Y.shape[0]
        square_distance = (euclidean_distance(X, Y))**2
        K = np.exp(-gamma*square_distance)
        diff = X[:, :, np.newaxis] - Y.T 
        K_gd = -2*gamma*diff*K[:, np.newaxis, :]
        return K_gd.reshape((N1*D, N))
    return rbf_gd

def rbfKernel_hess(gamma):
    def rbf_hess(X, Y=None):
        X, Y = check_array(X, Y)
        (N1, D), N = X.shape, Y.shape[0]
        square_distance = (euclidean_distance(X, Y))**2
        K = np.exp(-gamma*square_distance)
        K_hess = np.zeros((N1*D, N*D), dtype=np.complex64)
        E = np.eye(D, dtype=np.complex64)
        for i in range(0, N1*D, D):
            m = i//D
            for j in range(0, N*D, D):
                n = j//D
                diff = X[m] - Y[n] 
                K_hess[i:i+D, j:j+D] = (E - 2*gamma*diff[:, np.newaxis]*diff[np.newaxis, :])*K[m, n]
        K_hess *= 2*gamma
        return K_hess
    return rbf_hess

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
    assert ypred.ndim == ytest.ndim
    accuracy = 0
    for i, label in enumerate(ypred):
        if label == ytest[i]:
            accuracy += 1
    return accuracy*1.0 / n

def meanSquareError(ypred, ytest):
    assert ypred.ndim == ytest.ndim
    mse = np.mean((ypred - ytest)**2)
    return np.sqrt(mse)

def mean_abs_error(ypred, ytest):
    assert ypred.ndim == ytest.ndim
    err = np.mean(abs(ypred - ytest))
    return err

def max_abs_error(ypred, ytest):
    assert ypred.ndim == ytest.ndim
    err = np.amax(abs(ypred - ytest))
    return err