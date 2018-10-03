# SVR with mean square loss function + L2 norm

import numpy as np

def kernelMatrix(X, kernel):
    m = X.shape[0]
    KM = np.zeros((m, m))
    for i in range(m):
        KM[i, i] = kernel(X[i], X[i])
        for j in range(i+1, m):
            KM[i, j] = KM[j, i] = kernel(X[i], X[j])
    return KM

def svdSolver(X, y, kernelMatrix, Lambda):
    m = X.shape[0]
    A = kernelMatrix + Lambda*m*np.eye(m)
    U, S, Vh = np.linalg.svd(A)
    rank = len(S[S>1e-8])
    coefficients = np.divide((U.T[:rank] @ y), S[:rank])
    alpha = np.zeros(m)
    for j, cj in enumerate(coefficients):
        alpha += cj*Vh[j]
    return alpha

def value(Xtest, Xtrain, kernel, alpha):
    m, n = Xtrain.shape[0], Xtest.shape[0]
    predictValue = np.zeros(n)
    for i in range(n):
        KM = np.zeros(m)
        for j in range(m):
            KM[j] = kernel(Xtest[i], Xtrain[j])
        predictValue[i] = KM @ alpha
    return predictValue

# if __name__ == '__main__':

