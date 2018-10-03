# SVR api

import numpy as np 
from baseSVR import *

class Regressor(object):
    def __init__(self, kernel, Lambda):
        self.kernel = kernel
        self.Lambda_ = Lambda

    def fit(self, X, y):
        self.support_vector_ = X
        KM = kernelMatrix(X, self.kernel)
        alpha = svdSolver(X, y, KM, self.Lambda_)
        self.alpha_ = alpha
        return self

    def predict(self, X):
        predictValue = value(X, self.support_vector_, self.kernel, self.alpha_)
        return predictValue

if __name__ == '__main__':
    import pickle
    from sklearn.kernel_ridge import KernelRidge
    from default_kernels import *
    from metric import *

    fname = './dataset/toydataset/boston_data'
    with open(fname, 'rb') as f:
        bostonData = pickle.load(f)

    trainfeature, traintarget = bostonData[:300, :-1], bostonData[:300, -1]
    testfeature, testtarget = bostonData[300:, :-1], bostonData[300:, -1]
    bostonModel = Regressor(linearKernel, 1e-3)
    bostonModel.fit(trainfeature, traintarget)
    predictValue = bostonModel.predict(testfeature)
    precision = regressAccuracy(predictValue, testtarget)

    bostonModelKRR = KernelRidge(1e-3, kernel='linear')
    bostonModelKRR.fit(trainfeature, traintarget)
    predictValueKRR = bostonModelKRR.predict(testfeature)
    precisionKRR = regressAccuracy(predictValueKRR, testtarget)

    print(precision, precisionKRR)

