# common kernels

import numpy as np

def rbfKernel(gamma):
    def function(x, y):
        return np.exp(-gamma*np.sum((x - y)**2))
    return function

def polyKernel(gamma, r0, p):
    def function(x, y):
        return (r0 + gamma*(x @ y))**p
    return function

def linearKernel(x, y):
    return x @ y