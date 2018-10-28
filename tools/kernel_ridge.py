# KRR 

import numpy as np 
from ..main.regressor import baseRegressor
from ..main.utils import kernelMatrix, svd_solver

class KernelRidge(baseRegressor):
    def __init__(self, kernel, Lambda):
        super().__init__(kernel, Lambda, None)

    def fit(self, X, y, **kwargs):
        self.X_fit_ = X
        m = X.shape[0]
        KM = kernelMatrix(self.kernel, X)
        A = KM + self.Lambda_*m*np.eye(m)
        self.cond_ = np.linalg.cond(A)
        alpha = svd_solver(A, y, **kwargs)
        self.coef_ = alpha
        return self