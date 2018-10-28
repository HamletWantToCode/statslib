# SVC implementation
# with L2 norm

import numpy as np
from statslib.main.baseSVM import baseSVM





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

    kernel = rbfKernel(0.0125)
    optimizer = NAGMethod(0.1, 0.95, 1e-3, 1000)
    binary_model = hingeLossSVC(kernel, 1e-3, optimizer)
    binary_model.fit(b_trainfeatures, b_traintargets)
    b_predictlabels = binary_model.predict(b_testfeatures)
    modelPrecision = classifyAccuracy(b_predictlabels, b_testtargets)
    print(modelPrecision)

    # binary_modelSVC = SVC(1e3, kernel='rbf', gamma=0.01)
    # binary_modelSVC.fit(b_trainfeatures, b_traintargets)
    # b_predictlabelsSVC = binary_modelSVC.predict(b_testfeatures)
    # modelPrecision_SVC = classifyAccuracy(b_predictlabelsSVC, b_testtargets)

    # print(modelPrecision, modelPrecision_SVC)
