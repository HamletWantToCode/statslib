# support vector machine in classification and regression

import numpy as np 
from ..main.regressor import baseRegressor
from ..main.classifier import baseClassifier
from ..main.utils import kernelMatrix

class hingeLossSVC(baseClassifier):
    def lossFunction(self, alpha, X, y):
        m = X.shape[0]
        lossValue = 0
        KM = kernelMatrix(self.kernel, X, self.X_fit_)
        for i in range(m):
            lossValue += max([0, 1 - y[i]*(KM[i] @ alpha)])
        return (1.0/m)*lossValue + self.regularizer(alpha)

    def lossGradient(self, alpha, X, y):
        m, n = X.shape[0], self.X_fit_.shape[0]
        lossDerivativeValue = np.zeros(n)
        KM = kernelMatrix(self.kernel, X, self.X_fit_)
        for i in range(m):
            subgradient = 1 if y[i]*(KM[i] @ alpha) <= 1 else 0
            lossDerivativeValue += -y[i]*KM[i]*subgradient
        return (1.0/m)*lossDerivativeValue + self.regularizerGradient(alpha)

class EpsilonInsensitiveLossSVR(baseRegressor):
    def __init__(self, kernel, epsilon, Lambda, optimizer):
        super().__init__(kernel, Lambda, optimizer)
        self.epsilon_ = epsilon

    def lossFunction(self, alpha, X, y):
        m = X.shape[0]
        lossValue = 0
        KM = kernelMatrix(self.kernel, X, self.X_fit_)
        for i in range(m):
            z = y[i] - (KM[i] @ alpha)
            if abs(z) > self.epsilon_:
                lossValue += abs(z) - self.epsilon_
        regValue = self.regularizer(alpha)
        return (1.0/m)*lossValue + regValue

    def lossGradient(self, alpha, X, y):
        m, n = X.shape[0], self.X_fit_.shape[0]
        lossDeriv = np.zeros(n)
        KM = kernelMatrix(self.kernel, X, self.X_fit_)
        for i in range(m):
            z = y[i] - (KM[i] @ alpha)
            if z < -self.epsilon_:
                subgradient = -1
            elif abs(z) <= self.epsilon_:
                subgradient = 0
            else:
                subgradient = 1
            lossDeriv += -KM[i]*subgradient
        return (1.0/m)*lossDeriv + self.regularizerGradient(alpha)