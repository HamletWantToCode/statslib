# support vector machine in classification and regression

import numpy as np
from .base import baseClassifier, baseRegressor

# classification
class hingeLossSVC(baseClassifier):
    def __init__(self, kernel, Lambda, optimizer):
        super().__init__(kernel, Lambda, optimizer)

    def lossFunction(self, alpha, sub_KM, sub_y):
        n_samples = sub_KM.shape[0]
        Z = np.multiply(sub_y, sub_KM @ alpha)
        X = 1 - Z
        lossValue = 0
        for i in range(n_samples):
            lossValue += max(0, X[i])
        regular_term = self.regular_func(alpha)
        return (1.0/n_samples)*lossValue + 0.5*self.Lambda_*regular_term

    def lossGradient(self, alpha, sub_KM, sub_y):
        sub_y_column = sub_y.reshape((-1, 1))
        n_samples, n_coef = sub_KM.shape
        Z = np.multiply(sub_y, sub_KM @ alpha)
        dZ = np.multiply(sub_y_column, sub_KM)
        lossDerivativeValue = np.zeros(n_coef)
        for i in range(n_samples):
            subgradient = -1 if Z[i]<=1 else 0
            lossDerivativeValue += dZ[i]*subgradient
        regular_grad = self.regular_grad(alpha)
        return (1.0/n_samples)*lossDerivativeValue + self.Lambda_*regular_grad


# regression
class EpsilonInsensitiveLossSVR(baseRegressor):
    def __init__(self, kernel, epsilon, Lambda, optimizer):
        super().__init__(kernel, Lambda, optimizer)
        self.epsilon_ = epsilon

    def lossFunction(self, alpha, X, y):
        N_samples = X.shape[0]
        KM = baseRegressor._kernelMatrix(self.kernel, X, self.X_fit_)
        lossValue = 0
        for i in range(N_samples):
            fi = KM[i] @ alpha
            z = y[i] - fi
            if abs(z) > self.epsilon_:
                lossValue += abs(z) - self.epsilon_
        regular_term = baseRegressor._regularizer(self.kernel, alpha, self.X_fit_)
        return (1.0/N_samples)*lossValue + 0.5*self.Lambda_*regular_term

    def lossGradient(self, alpha, X, y):
        N_samples, N_X_fit = X.shape[0], self.X_fit_.shape[0]
        KM = baseRegressor._kernelMatrix(self.kernel, X, self.X_fit_)
        lossDeriv = np.zeros(N_X_fit)
        for i in range(N_samples):
            fi = KM[i] @ alpha
            z = y[i] - fi
            if z < -self.epsilon_:
                subgradient = -1
            elif abs(z) <= self.epsilon_:
                subgradient = 0
            else:
                subgradient = 1
            lossDeriv += -KM[i]*subgradient
        regular_grad = baseRegressor._regularizer_gradient(self.kernel, alpha, self.X_fit_)
        return (1.0/N_samples)*lossDeriv + self.Lambda_*regular_grad

