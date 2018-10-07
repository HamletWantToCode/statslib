# Multi Classes SVM

import numpy as np 
import copy

class OnevsRest(object):
    def __init__(self, baseEstimator):
        self.baseEstimator = baseEstimator

    def fit(self, X, y):
        labels = np.unique(y)
        numOfLabels = len(labels)
        self.labels_ = labels
        estimators = [copy.deepcopy(self.baseEstimator) for i in range(numOfLabels)]
        for i in range(numOfLabels):
            targets = np.zeros_like(y)
            targets[y==labels[i]] = 1
            targets[y!=labels[i]] = -1
            (estimators[i]).fit(X, targets)
        self.estimators_ = estimators
        return self

    def predict(self, X):
        n, m = X.shape[0], self.labels_.shape[0]
        maxima = np.empty(n, dtype=float)
        maxima.fill(-np.inf)
        argmaxima = np.zeros(n, dtype=int)
        for i, estimator in enumerate(self.estimators_):
            label = estimator.predict(X)
            np.maximum(maxima, label, out=maxima)
            argmaxima[maxima == label] = i
        predictLabels = self.labels_[argmaxima]
        return predictLabels

if __name__ == '__main__':
    from classifier import *
    from main.default_kernels import *
    from main.default_optimizers import *
    import pickle
    from main.metric import *

    with open('./toydataset/iris_data', 'rb') as f:
        Data = pickle.load(f)

    Feature, Target = Data[:, :-1], Data[:, -1]
    meanOfFeature = np.mean(Feature, axis=0, keepdims=True)
    Feature -= meanOfFeature
    trainFeature, trainTarget = Feature[:100], Target[:100]
    testFeature, testTarget = Feature[100:], Target[100:]

    covariance = trainFeature.T @ trainFeature
    U, S, Vt = np.linalg.svd(covariance)
    decomposedTrainFeature = (Vt[:2] @ trainFeature.T).T
    decomposedTestFeature = (Vt[:2] @ testFeature.T).T

    kernel = rbfKernel(0.01)
    optimizer = NAGMethod(0.1, 0.95, 1e-3, 1000)
    estimator = hingeLossSVC(kernel, 1e-3, optimizer)
    model = OnevsRest(estimator)
    model.fit(decomposedTrainFeature, trainTarget)
    predictlabels = model.predict(decomposedTestFeature)
    print(predictlabels)
    print(testTarget)

    precision = classifyAccuracy(predictlabels, testTarget)
    print(precision)