# regressor

import numpy as np
from .baseKM import baseKernelMachine
from .utils import kernelMatrix

class baseRegressor(baseKernelMachine):
    def __init__(self, kernel, Lambda, optimizor):
        super().__init__(kernel)
        self.Lambda_ = Lambda
        self.optimizor = optimizor

    def regularizer(self, alpha):
        KM = kernelMatrix(self.kernel, self.X_fit_)
        return 0.5*self.Lambda_*(alpha @ (KM @ alpha))

    def regularizerGradient(self, alpha):
        KM = kernelMatrix(self.kernel, self.X_fit_)
        return self.Lambda_*(KM @ alpha)

    def fit(self, X, y):
        m = X.shape[0]
        self.X_fit_ = X
        alpha0 = np.zeros(m)
        function, gradient = self.lossFunction, self.lossGradient
        alpha = self.optimizor.run(alpha0, function, gradient, X, y)
        self.coef_ = alpha
        return self

    def predict(self, X):
        return self.decisionFunction(self.coef_, X)