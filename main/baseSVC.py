# support vector machine
# 2 class classifier: hinge loss + L2 regularization

import numpy as np 

def kernelMatrix(X, kernel):
    m = X.shape[0]
    KM = np.zeros((m, m))
    for i in range(m):
        KM[i, i] = kernel(X[i], X[i])
        for j in range(i+1, m):
            KM[i, j] = KM[j, i] = kernel(X[i], X[j])
    return KM

def lossFunction(X, y, kernelMatrix, Lambda):
    m = X.shape[0]
    def hingeLoss(alpha):
        lossValue = 0
        for i in range(m):
            lossValue += max([0, 0.9 - y[i]*(kernelMatrix[i] @ alpha)])
        return (1.0/m)*lossValue + 0.5*Lambda*(alpha @ (kernelMatrix @ alpha))
    return hingeLoss

def lossFunctionDerivative(X, y, kernelMatrix, Lambda):
    m = X.shape[0]
    def hingeLossDerivative(alpha):
        lossDerivativeValue = np.zeros(m)
        for i in range(m):
            subgradient = 1 if y[i]*(kernelMatrix[i] @ alpha) < 1 else 0
            lossDerivativeValue += -y[i]*kernelMatrix[i]*subgradient
        return (1.0/m)*lossDerivativeValue + Lambda*(kernelMatrix @ alpha)
    return hingeLossDerivative

def gradientDescent(alpha0, function, derivative, step, stopError=1e-3, maxiter=10000):
    i = 1
    while True:
        alpha = alpha0 - step*derivative(alpha0)
        # if i%10 == 0:
        #     print(function(alpha) - function(alpha0))
        if abs(function(alpha) - function(alpha0)) < stopError:
            alpha0 = alpha
            print('optimization converge after %s steps!' %(i))
            break
        elif i > maxiter:
            raise ValueError('loops exceed the maximum iteration !')
        alpha0 = alpha
        i += 1
    return alpha

def distance(Xtrain, Xtest, alpha, kernel):
    m, n = Xtrain.shape[0], Xtest.shape[0]
    displacement = np.zeros(n)
    for i in range(n):
        KM = np.zeros(m)
        for j in range(m):
            KM[j] = kernel(Xtest[i], Xtrain[j])
        displacement[i] = KM @ alpha
    return displacement
            
if __name__ == '__main__':
    import pickle
    from default_kernels import *
    from sklearn.svm import SVC

    fname = './dataset/toydataset/Gaussian_data'
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    # Target = data[:, -1]; Target[Target==0] = -1 # for breast cancer database !
    trainSize = 400
    trainFeature, trainTarget = data[:trainSize, :-1], data[:trainSize, -1]
    testFeature, testTarget = data[trainSize:, :-1], data[trainSize:, -1]

    KM = kernelMatrix(trainFeature, linearKernel)
    hingeLoss = lossFunction(trainFeature, trainTarget, KM, Lambda=1e-4)
    hingeLossDerivative = lossFunctionDerivative(trainFeature, trainTarget, KM, Lambda=1e-4)
    alpha0 = np.zeros(trainFeature.shape[0])
    alphaOptimal = gradientDescent(alpha0, hingeLoss, hingeLossDerivative, 0.1, 1e-3)
    testSampleDistance = distance(trainFeature, testFeature, alphaOptimal, linearKernel)
    testDataSize = testSampleDistance.shape[0]
    predictedLabel = np.zeros(testDataSize)
    predictedLabel = np.array([1 if d>0 else -1 for d in testSampleDistance])
    accuracy = 0
    for ix, label in enumerate(predictedLabel):
        if label == testTarget[ix]:
            accuracy += 1
    precision = accuracy*1.0 / testDataSize
    print(precision)

    model = SVC(C=1e4, kernel='linear')
    model.fit(trainFeature, trainTarget)
    predictLabel_sklearn = model.predict(testFeature)
    difference_sklearn = np.sum(abs(predictLabel_sklearn - testTarget))
    precision_sklearn = (testDataSize - difference_sklearn)*1.0 / testDataSize
    print(precision_sklearn)

    import matplotlib.pyplot as plt 
    plt.scatter(testFeature[:, 0], testFeature[:, 1], c=testTarget)
    decisionFunction = lambda x: -(1.0/(trainFeature[:, 1] @ alphaOptimal))*(trainFeature[:, 0] @ alphaOptimal)*x
    decisionFunction_sklearn = lambda x: -(1.0/model.coef_[0][1])*(model.coef_[0][0]*x + model.intercept_)

    X = np.linspace(-5, 5, 50)
    y = decisionFunction(X)
    y_sklearn = decisionFunction_sklearn(X)
    plt.plot(X, y, 'r')
    plt.plot(X, y_sklearn, 'g')
    plt.show()
    
