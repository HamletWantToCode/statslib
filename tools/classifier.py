# SVC implementation

import numpy as np 
from main.baseSVM import baseSVM

class hingeLossSVC(baseSVM):
    def lossFunction(self, X, y, KM):
        m = X.shape[0]
        def hingeLoss(alpha):
            lossValue = 0
            for i in range(m):
                lossValue += max([0, 1 - y[i]*(KM[i] @ alpha)])
            return (1.0/m)*lossValue + 0.5*self.Lambda_*(alpha @ (KM @ alpha))
        return hingeLoss

    def lossDerivative(self, X, y, KM):
        m = X.shape[0]
        def hingeLossDerivative(alpha):
            lossDerivativeValue = np.zeros(m)
            for i in range(m):
                for j in range(m):
                    subgradient = 1 if y[j]*(KM[j] @ alpha) <= 1 else 0
                    lossDerivativeValue[i] += -y[j]*KM[i, j]*subgradient
            return (1.0/m)*lossDerivativeValue + self.Lambda_*(KM @ alpha)
        return hingeLossDerivative

    def predict(self, X):
        n = X.shape[0]
        distance = self.decisionFunction(X)
        predictLabels = np.zeros(n)
        for i in range(n):
            predictLabels[i] = 1 if distance[i]>0 else -1
        return predictLabels

if __name__ == '__main__':
    import pickle
    from main.default_kernels import *
    from sklearn.svm import SVC
    from main.metric import *
    from main.default_optimizers import *

    # binary classification
    fname_binary = '../toydataset/Breast_cancer_data'
    with open(fname_binary, 'rb') as f:
        breast_cancer_data = pickle.load(f)
    b_trainfeatures, b_traintargets = breast_cancer_data[:300, :-1], breast_cancer_data[:300, -1]
    b_testfeatures, b_testtargets = breast_cancer_data[300:, :-1], breast_cancer_data[300:, -1]
    numOfTestData = b_testfeatures.shape[0]

    optimizer = GradientDescent(0.1, 1e-3, 1000)
    binary_model = hingeLossSVC(rbfKernel, 1e-3, optimizer)
    binary_model.fit(b_trainfeatures, b_traintargets)
    b_predictlabels = binary_model.predict(b_testfeatures)
    modelPrecision = classifyAccuracy(b_predictlabels, b_testtargets)

    binary_modelSVC = SVC(1e3, kernel='rbf', gamma=0.01)
    binary_modelSVC.fit(b_trainfeatures, b_traintargets)
    b_predictlabelsSVC = binary_modelSVC.predict(b_testfeatures)
    modelPrecision_SVC = classifyAccuracy(b_predictlabelsSVC, b_testtargets)

    print(modelPrecision, modelPrecision_SVC)    