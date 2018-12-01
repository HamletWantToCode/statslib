# multi-task KRR

import numpy as np
from statslib.main.base import BaseRegressor
from statslib.tools.utils import svd_solver

class Multi_task_KRR(BaseRegressor):
    def __init__(self, kernel, kernel_gd, Lambda):
        super().__init__(kernel, Lambda, None)
        self.kernel_gd = kernel_gd

    def fit(self, X, y, dy, cond=1e-8):
        assert (X.ndim==2) & (y.ndim==2), print('reshape array into 2 dimension !')
        n_samples = X.shape[0]
        self.X_fit_ = X
        K = self.kernel(X)
        M = self.kernel_gd(X)
        A = 0.93*K @ (K + self.Lambda_*n_samples*np.eye(n_samples)) + 0.07*np.trace(M[:, :, :, np.newaxis]*M[:, :, np.newaxis, :], axis1=0, axis2=1)
        b = 0.93*K @ y + 0.07*np.trace(M*dy[:, :, np.newaxis], axis1=0, axis2=1)[:, np.newaxis]
        alpha = svd_solver(A, b, cond)
        self.coef_ = alpha
        return self
