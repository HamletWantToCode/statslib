# common kernels

import numpy as np

def rbfKernel(x, y):
    gamma = 0.01
    return np.exp(-gamma*np.sqrt(np.sum((x - y)**2)))

def linearKernel(x, y):
    return x @ y