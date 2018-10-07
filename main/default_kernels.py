# common kernels

import numpy as np

def rbfKernel(gamma):
    def function(x, y):
        return np.exp(-gamma*np.sqrt(np.sum((x - y)**2)))
    return function

def linearKernel(x, y):
    return x @ y