# KRR 

import numpy as np 
from ..main.kernel_machine import baseRegressor

class KernelRidge(baseRegressor):
    @staticmethod
    def _svd_solver(A, b, cond=None):
        U, S, Vh = np.linalg.svd(A)
        if cond is None:
            cond = np.finfo(np.float64).eps
        rank = len(S[S>cond])
        coefficients = np.divide((U.T[:rank] @ b), S[:rank])
        x = np.zeros(A.shape[1])
        for j, cj in enumerate(coefficients):
            x += cj*Vh[j]
        return x

    def __init__(self, kernel, Lambda):
        super().__init__(kernel, Lambda, None)

    def fit(self, X, y, cond=1e-8):
        N_samples = X.shape[0]
        self.X_fit_ = X
        KM = baseRegressor._kernelMatrix(self.kernel, X)
        A = KM + self.Lambda_*N_samples*np.eye(N_samples)
        self.cond_ = np.linalg.cond(A)
        alpha = KernelRidge._svd_solver(A, y, cond)
        self.coef_ = alpha
        return self


    