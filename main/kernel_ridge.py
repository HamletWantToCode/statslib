# KRR 

import numpy as np 
from ..main.base import BaseRegressor
from ..tools.utils import svd_solver

class KernelRidge(BaseRegressor):
    def __init__(self, kernel, Lambda):
        super().__init__(kernel, Lambda, None)

    def fit(self, X, y, cond=1e-8):
        assert (X.ndim==2) & (y.ndim==2), print('reshape array into 2 dimension !')
        n_samples = X.shape[0]
        self.X_fit_ = X
        KM = self.kernel(X)
        A = KM + self.Lambda_*n_samples*np.eye(n_samples)
        self.cond_ = np.linalg.cond(A)
        alpha = svd_solver(A, y, cond)
        self.coef_ = alpha
        return self


    