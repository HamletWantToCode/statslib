# kernel method base class

import numpy as np 
from ..tools.utils import load_data, regularizer, regularizer_gradient

# kernel machine 
class BaseKernelMachine(object):
    def __init__(self, kernel, Lambda, optimizer):
        self.kernel = kernel
        self.Lambda_ = Lambda
        self.optimizer = optimizer

    def lossFunction(self):
        raise NotImplementedError('MUST implement loss function before using fit !')

    def lossGradient(self):
        raise NotImplementedError('MUST implement loss gradient before using fit !')
    
    def fit(self, X, y):
        n_sample = X.shape[0]
        self.X_fit_ = X
        full_KM = self.kernel(X)
        self.regular_func = regularizer(full_KM)
        self.regular_grad = regularizer_gradient(full_KM)
        # alpha0 = np.zeros(n_sample)
        alpha0 = np.random.uniform(-1, 1, n_sample)*1e-2
        alpha = self.optimizer.run(alpha0, self.lossFunction, self.lossGradient, full_KM, y)
        self.coef_ = alpha
        return self

    def decisionFunction(self, X):
        kernel_vector = self.kernel(X, self.X_fit_)
        return kernel_vector @ self.coef_

class BaseClassifier(BaseKernelMachine):
    def predict(self, X):
        n = X.shape[0]
        distance = self.decisionFunction(X)
        predictLabels = np.zeros(n)
        for i in range(n):
            predictLabels[i] = 1 if distance[i]>0 else -1
        return predictLabels

class BaseRegressor(BaseKernelMachine):
    def predict(self, X):
        return self.decisionFunction(X)

# optimization
class BaseOptimize(object):
    def __init__(self, learning_rate, stopError, maxiters, n_batch=1, verbose=0):
        self.lr_ = learning_rate
        self.stopError_ = stopError
        self.maxiters_ = maxiters
        self.nb_ = n_batch
        self.verbose_ = verbose
        if verbose:
            self.fvals_ = []

    def run(self, alpha, function, gradient, full_KM, y):
        data = np.c_[full_KM, y]
        f0 = function(alpha, full_KM, y)
        n_epoch = 0
        while True:
            loader = load_data(data, self.nb_)
            for i, (sub_KM, sub_y) in enumerate(loader):
                alpha = self.optimizer(alpha, gradient, sub_KM, sub_y)
                if self.verbose_:
                    f_update = function(alpha, full_KM, y)
                    self.fvals_.append(f_update)
            f1 = function(alpha, full_KM, y)
            ferr = abs(f1-f0)
            if ferr < self.stopError_:
                print('optimization converges after %d epoches and %d batch iterations!' %(n_epoch, i+1))
                break
            f0 = f1
            if n_epoch > self.maxiters_:
                raise StopIteration('loop exceeds the maximum iterations !')
            n_epoch += 1
        return alpha


