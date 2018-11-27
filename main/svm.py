# support vector machine in classification and regression

import numpy as np
from .base import BaseClassifier, BaseRegressor

# classification
class hingeLossSVC(BaseClassifier):
    def __init__(self, kernel, Lambda, optimizer):
        super().__init__(kernel, Lambda, optimizer)

    def lossFunction(self, alpha, sub_KM, sub_y):
        n_samples = sub_KM.shape[0]
        Z = np.squeeze(sub_y)*(sub_KM @ alpha)
        X = 1 - Z
        lossValue = np.sum(X[X>0])
        regular_term = self.regular_func(alpha)
        return (1.0/n_samples)*lossValue + 0.5*self.Lambda_*regular_term

    def lossGradient(self, alpha, sub_KM, sub_y):
        n_samples = sub_KM.shape[0]
        dZ = sub_y*sub_KM
        Z = sub_y*(sub_KM @ alpha[:, np.newaxis])
        subgradient = np.where(Z>1, 0, -1)
        lossDerivativeValue = np.sum(subgradient*dZ, axis=0)
        regular_term_grad = self.regular_grad(alpha)
        return (1.0/n_samples)*lossDerivativeValue + self.Lambda_*regular_term_grad


# regression
class EpsilonInsensitiveLossSVR(BaseRegressor):
    def __init__(self, kernel, Lambda, optimizer, epsilon):
        super().__init__(kernel, Lambda, optimizer)
        self.epsilon_ = epsilon

    def lossFunction(self, alpha, sub_KM, sub_y):
        n_samples = sub_KM.shape[0]
        abs_Z = abs(np.squeeze(sub_y) - (sub_KM @ alpha))
        abs_Z -= self.epsilon_
        lossValue = np.sum(abs_Z[abs_Z>0])
        regular_term = self.regular_func(alpha)
        return (1.0/n_samples)*lossValue + 0.5*self.Lambda_*regular_term

    def lossGradient(self, alpha, sub_KM, sub_y):
        n_samples = sub_KM.shape[0]
        Z = sub_y - (sub_KM @ alpha[:, np.newaxis])
        subgradient = np.where(Z>self.epsilon_, -1, 0)
        subgradient[Z<=-self.epsilon_] = 1
        lossDeriv = np.sum(subgradient*sub_KM, axis=0)
        regular_term_grad = self.regular_grad(alpha)
        return (1.0/n_samples)*lossDeriv + 0.5*self.Lambda_*regular_term_grad

