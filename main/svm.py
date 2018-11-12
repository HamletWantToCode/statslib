# support vector machine in classification and regression

import numpy as np
from .base import BaseClassifier, BaseRegressor

# classification
class hingeLossSVC(BaseClassifier):
    def __init__(self, kernel, Lambda, optimizer):
        super().__init__(kernel, Lambda, optimizer)

    def lossFunction(self, alpha, sub_KM, sub_y):
        n_samples = sub_KM.shape[0]
        Z = sub_y*(sub_KM @ alpha)
        X = 1 - Z
        lossValue = 0
        for i in range(n_samples):
            lossValue += max(0, X[i])
        regular_term = self.regular_func(alpha)
        return (1.0/n_samples)*lossValue + 0.5*self.Lambda_*regular_term

    def lossGradient(self, alpha, sub_KM, sub_y):
        sub_y_column = sub_y[:, np.newaxis]
        n_samples, n_coef = sub_KM.shape
        Z = sub_y*(sub_KM @ alpha)
        dZ = sub_y_column*sub_KM
        lossDerivativeValue = np.zeros(n_coef)
        for i in range(n_samples):
            subgradient = -1 if Z[i]<=1 else 0
            lossDerivativeValue += dZ[i]*subgradient
        regular_term_grad = self.regular_grad(alpha)
        return (1.0/n_samples)*lossDerivativeValue + self.Lambda_*regular_term_grad


# regression
class EpsilonInsensitiveLossSVR(BaseRegressor):
    def __init__(self, kernel, Lambda, optimizer, epsilon):
        super().__init__(kernel, Lambda, optimizer)
        self.epsilon_ = epsilon

    def lossFunction(self, alpha, sub_KM, sub_y):
        n_samples = sub_KM.shape[0]
        abs_Z = abs(sub_y - (sub_KM @ alpha))
        abs_Z -= self.epsilon_
        lossValue = 0
        for i in range(n_samples):
            lossValue += max(0, abs_Z[i])
        regular_term = self.regular_func(alpha)
        return (1.0/n_samples)*lossValue + 0.5*self.Lambda_*regular_term

    def lossGradient(self, alpha, sub_KM, sub_y):
        n_samples, n_coef = sub_KM.shape
        Z = sub_y - (sub_KM @ alpha)
        lossDeriv = np.zeros(n_coef)
        for i in range(n_samples):
            if Z[i] <= -self.epsilon_:
                subgradient = 1
            elif Z[i] > self.epsilon_:
                subgradient = -1
            else:
                subgradient = 0
            lossDeriv += subgradient*sub_KM[i]
        regular_term_grad = self.regular_grad(alpha)
        return (1.0/n_samples)*lossDeriv + 0.5*self.Lambda_*regular_term_grad

