# miscellanious utility

import numpy as np 

# data
def kernelMatrix(kernel, X, Y=None):
    if Y is None:
        m = n = X.shape[0]
        Y = X
    else:
        m, n = X.shape[0], Y.shape[0]
    KM = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            KM[i, j] = kernel(X[i], Y[j])
    return KM

def data_spliter(data, nsplits):
    np.random.shuffle(data)
    N = data.shape[0]
    num_pieces = N // nsplits
    for i in range(1, nsplits+1):
        yield data[(i-1)*num_pieces:i*num_pieces]

# linear system solver
def svd_solver(A, b, cond=None):
    U, S, Vh = np.linalg.svd(A)
    if cond is None:
        cond = np.finfo(np.float64).eps
    rank = len(S[S>cond])
    coefficients = np.divide((U.T[:rank] @ b), S[:rank])
    x = np.zeros(A.shape[1])
    for j, cj in enumerate(coefficients):
        x += cj*Vh[j]
    return x

# metric
def classifyAccuracy(ypred, ytest):
    n = len(ytest)
    assert len(ypred) == n
    accuracy = 0
    for i, label in enumerate(ypred):
        if label == ytest[i]:
            accuracy += 1
    return accuracy*1.0 / n

def meanSquareError(ypred, ytest):
    n = len(ytest)
    assert len(ypred) == n
    mse = np.sqrt(np.sum((ypred - ytest)**2)) / n
    return mse