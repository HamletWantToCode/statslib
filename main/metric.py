# precision measure

import numpy as np 

def classifyAccuracy(ypred, ytest):
    n = len(ytest)
    assert len(ypred) == n
    accuracy = 0
    for i, label in enumerate(ypred):
        if label == ytest[i]:
            accuracy += 1
    return accuracy*1.0 / n
    
def regressAccuracy(ypred, ytest):
    n = len(ytest)
    assert len(ypred) == n
    mse = np.sqrt(np.sum((ypred - ytest)**2)) / n
    return mse