# SVR implementation
# with L2 norm

import numpy as np
from statslib.main.baseSVM import baseSVM
from statslib.main.default_linearEquationSolver import svd_solver

class SquareLossSVR(baseSVM):
    def __init__(self, kernel, Lambda):
        super().__init__(kernel, Lambda, None)

    def fit(self, X, y, cond=-1):
        self.support_vectors_ = X
        m = X.shape[0]
        KM = self.kernelMatrix()
        A = KM + self.Lambda_*m*np.eye(m)
        self.cond_ = np.linalg.cond(A)
        # if self.cond_ < 1e3:
        #     alpha = svd_solver(A, y, cond)
        # else:
        #     alpha = RileyGolub_solver(A, y)
        alpha = svd_solver(A, y, cond)
        self.coef_ = alpha
        return self

    def predict(self, X):
        return self.decisionFunction(X)

class EpsilonInsensitiveLossSVR(baseSVM):
    def __init__(self, kernel, epsilon, Lambda, optimizer):
        super().__init__(kernel, Lambda, optimizer)
        self.epsilon_ = epsilon

    def lossFunction(self, X, y, KM):
        m = X.shape[0]
        def function(alpha):
            lossValue = 0
            for i in range(m):
                z = y[i] - (KM[i] @ alpha)
                if abs(z) > self.epsilon_:
                    lossValue += abs(z) - self.epsilon_
            return (1.0/m)*lossValue + 0.5*self.Lambda_*(alpha @ (KM @ alpha))
        return function

    def lossDerivative(self, X, y, KM):
        m = X.shape[0]
        def function(alpha):
            lossDeriv = np.zeros(m)
            for i in range(m):
                for j in range(m):
                    z = y[j] - (KM[j] @ alpha)
                    if z < -self.epsilon_:
                        subgradient = -1
                    elif abs(z) <= self.epsilon_:
                        subgradient = 0
                    else:
                        subgradient = 1
                    lossDeriv[i] += -KM[j, i]*subgradient
            return (1.0/m)*lossDeriv + self.Lambda_*(KM @ alpha)
        return function

    def predict(self, X):
        return self.decisionFunction(X)

if __name__ == '__main__':
    import pickle
    from sklearn.kernel_ridge import KernelRidge
    # from sklearn.preprocessing import StandardScaler
    from main.default_kernels import *
    from main.metric import *
    from main.default_optimizers import *

    fname = '../toydataset/boston_data'
    with open(fname, 'rb') as f:
        bostonData = pickle.load(f)

    trainfeature, traintarget = bostonData[:300, :-1], bostonData[:300, -1]
    testfeature, testtarget = bostonData[300:, :-1], bostonData[300:, -1]
    # stdScaler = StandardScaler()
    # newtrainfeature = stdScaler.fit_transform(trainfeature)

    optimizer = GradientDescent(0.1, 1e-3, 1000)
    bostonModel = SquareLossSVR(linearKernel, 1e-3)
    bostonModel.fit(trainfeature, traintarget)
    predictValue = bostonModel.predict(testfeature)
    precision = regressAccuracy(predictValue, testtarget)

    bostonModelKRR = KernelRidge(1e-3, kernel='linear')
    bostonModelKRR.fit(trainfeature, traintarget)
    predictValueKRR = bostonModelKRR.predict(testfeature)
    precisionKRR = regressAccuracy(predictValueKRR, testtarget)

    print(precision, precisionKRR)
