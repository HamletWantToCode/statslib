# SVM base class

import numpy as np 

from .utils import kernelMatrix

class baseKernelMachine(object):
    def __init__(self, kernel):
        self.kernel = kernel

    def decisionFunction(self, alpha, X):
        KM = kernelMatrix(self.kernel, X, self.X_fit_)
        decisionValue = KM @ alpha
        return decisionValue

