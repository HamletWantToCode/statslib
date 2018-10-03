# multilabel classifier, ensemble learning

import numpy as np
from baseSVC import *

class Classifier(object):
    def __init__(self, kernel, Lambda, step, tolr=1e-3):
        self.kernel = kernel
        self.Lambda_ = Lambda
        self.step_ = step
        self.tolr_ = tolr

    def fit(self, X, y):
        self.support_vector_ = X
        KM = kernelMatrix(X, self.kernel)
        alpha0 = np.zeros(X.shape[0])
        hingeLoss = lossFunction(X, y, KM, self.Lambda_)
        hingeLossDerivative = lossFunctionDerivative(X, y, KM, self.Lambda_)
        alpha = gradientDescent(alpha0, hingeLoss, hingeLossDerivative, self.step_, self.tolr_)
        self.alpha_ = alpha
        return self
    
    def decision_function(self, X):
        return distance(self.support_vector_, X, self.alpha_, self.kernel)
    
    def predict(self, X):
        n = X.shape[0]
        distance = self.decision_function(X)
        predictLabels = np.zeros(n)
        for i in range(n):
            predictLabels[i] = 1 if distance[i]>0 else -1
        return predictLabels

class MultiClassOvR(object):
    """
    choose a label based on the maximum distance 
    """
    def __init__(self, classes, baseEstimator, *args):
        self.baseEstimator = baseEstimator
        self.classes_ = classes
        self.args = args
    
    def fit(self, X, y):
        ListOfEstimators = []
        m = X.shape[0]
        for i, label in enumerate(self.classes_):
            Targets = np.zeros(m)
            Targets[y==label] = 1
            Targets[y!=label] = -1
            estimator = self.baseEstimator(*self.args)
            estimator.fit(X, Targets)
            ListOfEstimators.append(estimator)
        self.multiEstimator = ListOfEstimators
        return self
    
    def predict(self, X):
        n, m = X.shape[0], len(self.classes_)
        maxima = np.empty(n, dtype=float)
        maxima.fill(-np.inf)
        argmaxima = np.zeros(n, dtype=int)
        for i, estimator in enumerate(self.multiEstimator):
            label = estimator.predict(X)
            np.maximum(maxima, label, out=maxima)
            argmaxima[maxima == label] = i
        predictLabels = self.classes_[argmaxima]
        return predictLabels

if __name__ == '__main__':
    import pickle
    from default_kernels import *
    from sklearn.svm import SVC
    from metric import *

    # binary classification
    # fname_binary = './dataset/toydataset/Breast_cancer_data'
    # with open(fname_binary, 'rb') as f:
    #     breast_cancer_data = pickle.load(f)
    # b_trainfeatures, b_traintargets = breast_cancer_data[:300, :-1], breast_cancer_data[:300, -1]
    # b_testfeatures, b_testtargets = breast_cancer_data[300:, :-1], breast_cancer_data[300:, -1]
    # numOfTestData = b_testfeatures.shape[0]
    # binary_model = Classifier(rbfKernel, 1e-3, 1)
    # binary_model.fit(b_trainfeatures, b_traintargets)
    # b_predictlabels = binary_model.predict(b_testfeatures)
    # modelPrecision = classifyAccuracy(b_predictlabels, b_testtargets)

    # binary_modelSVC = SVC(1e3, kernel='rbf', gamma=0.01)
    # binary_modelSVC.fit(b_trainfeatures, b_traintargets)
    # b_predictlabelsSVC = binary_modelSVC.predict(b_testfeatures)
    # modelPrecision_SVC = np.sum(abs(b_predictlabelsSVC - b_testtargets))*1.0 / numOfTestData

    # print(modelPrecision, modelPrecision_SVC)

    # three-class classification
    fname_triple = './dataset/toydataset/wine_data'
    with open(fname_triple, 'rb') as f:
        wine_data = pickle.load(f)
    t_trainfeatures, t_traintargets = wine_data[:80, :-1], wine_data[:80, -1]
    t_testfeatures, t_testtargets = wine_data[80:, :-1], wine_data[80:, -1]
    numOfTestData = t_testfeatures.shape[0]
    triple_model = MultiClassOvR(np.array([0, 1, 2]), Classifier, rbfKernel, 1e-4, 1)
    triple_model.fit(t_trainfeatures, t_traintargets)
    t_predictlabels = triple_model.predict(t_testfeatures)
    modelPrecision = classifyAccuracy(t_predictlabels, t_testtargets)

    triple_modelSVC = SVC(1e3, kernel='rbf', gamma=0.01)
    triple_modelSVC.fit(t_trainfeatures, t_traintargets)
    t_predictlabelsSVC = triple_modelSVC.predict(t_testfeatures)
    modelPrecision_SVC = classifyAccuracy(t_predictlabelsSVC, t_testtargets)

    print(modelPrecision, modelPrecision_SVC)
    

