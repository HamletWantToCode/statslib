# KRR 

import numpy as np 
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y, check_array

from statslib.main.utils import cholesky_solver, rbf_kernel, rbf_kernel_gradient
class KernelRidge(BaseEstimator, RegressorMixin):
    """ An customized implementation of kernel ridge regression.

    Parameters
    ----------
    gamma: 
    C:
    kernel: callable

    Attributes
    ----------
    X_fit_:
    cond_:
    coef_:
    """
    def __init__(self, C=1e-10, gamma=1e-3, kernel=rbf_kernel, kernel_gd=rbf_kernel_gradient):
        self.gamma = gamma
        self.C = C
        self.kernel = kernel
        self.kernel_gd = kernel_gd

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        self.X_fit_ = X
        n_dims = X.shape[0]
        A = self.kernel(self.gamma, X, X) + self.C*n_dims*np.eye(n_dims)
        self.cond_ = np.linalg.cond(A)
        self.coef_ = cholesky_solver(A, y, k=0)
        return self

    def predict(self, X):
        """ 
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
        """
        X = check_array(X)
        y_pred = self.kernel(self.gamma, X, self.X_fit_) @ self.coef_
        return y_pred

    def predict_gradient(self, X):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples, n_features) 
        """
        X = check_array(X)
        dy_pred = self.kernel_gd(self.gamma, X, self.X_fit_) @ self.coef_
        return dy_pred
    