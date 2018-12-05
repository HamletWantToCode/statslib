# Gauss Process Regression

import numpy as np 
from statslib.tools.utils import svd_solver

class Gauss_Process_Regressor(object):
    def __init__(self, kernel, sigma, kernel_gd=None, kernel_hess=None):
        self.kernel = kernel
        self.sigma_ = sigma
        self.kernel_gd = kernel_gd
        self.kernel_hess = kernel_hess

    def fit(self, X, y, cond=None):
        assert X.ndim==2 and y.ndim==2, print('dimension not match')
        self.X_fit_ = X
        Cov = cov_setup(X, self.kernel, self.kernel_gd, self.kernel_hess)
        A = Cov + (self.sigma_**2)*np.eye(Cov.shape[0])
        self.coef_ = svd_solver(A, y, cond)
        return self

    def predict(self, X):
        K_star = kstar_setup(X, self.X_fit_, self.kernel, self.kernel_gd)
        predict_y = K_star @ self.coef_
        K_predict = self.kernel(X, X)
        Cov = cov_setup(self.X_fit_, self.kernel, self.kernel_gd, self.kernel_hess)
        A = Cov + (self.sigma_**2)*np.eye(Cov.shape[0])
        U, S, Vh = np.linalg.svd(A)
        rank = len(S)
        A_inv = (Vh[:rank].T).conj() @ np.diag(1./S) @ U[:, :rank].T
        Err = K_predict - K_star @ A_inv @ K_star.T
        np.maximum(0, Err, out=Err)
        return predict_y, np.sqrt(np.diag(Err))

def cov_setup(X, func, dfunc=None, ddfunc=None):
    K = func(X)
    if (dfunc is not None) & (ddfunc is not None):
        K_gd = dfunc(X)
        K_hess = ddfunc(X)
        tmp1, tmp2 = np.r_[K, K_gd], np.r_[K_gd.T, K_hess]
        G = np.c_[tmp1, tmp2]
        return G
    else:
        return K

def kstar_setup(X, Y, func, dfunc=None):
    Kstar = func(X, Y)
    if dfunc is not None:
        Kstar_gd = dfunc(X, Y)
        Gstar = np.c_[Kstar, -Kstar_gd]
        return Gstar
    else:
        return Kstar

