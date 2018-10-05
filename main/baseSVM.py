# SVM base class

import numpy as np 

class baseSVM(object):
    def __init__(self, kernel, Lambda, optimizer):
        self.kernel = kernel
        self.Lambda_ = Lambda
        self.optimizer = optimizer
    
    def kernelMatrix(self):
        m = self.support_vectors_.shape[0]
        KM = np.zeros((m, m))
        for i in range(m):
            KM[i, i] = self.kernel(self.support_vectors_[i], self.support_vectors_[i])
            for j in range(i+1, m):
                KM[i, j] = KM[j, i] = self.kernel(self.support_vectors_[i], self.support_vectors_[j])
        return KM

    def lossFunction(self, X, y, KM):
        pass

    def lossDerivative(self, X, y, KM):
        pass

    def fit(self, X, y):
        self.support_vectors_ = X
        KM = self.kernelMatrix()
        Loss = self.lossFunction(X, y, KM)
        LossDerivative = self.lossDerivative(X, y, KM)
        alpha0 = np.zeros(X.shape[0])
        self.optimizer.run(alpha0, Loss, LossDerivative, debug=False)
        self.coef_ = self.optimizer.alpha_
        return self

    def decisionFunction(self, X):
        m, n = self.support_vectors_.shape[0], X.shape[0]
        decisionValue = np.zeros(n)
        for i in range(n):
            KM = np.zeros(m)
            for j in range(m):
                KM[j] = self.kernel(X[i], self.support_vectors_[j])
            decisionValue[i] = KM @ self.coef_
        return decisionValue

    def predict(self, X):
        pass

