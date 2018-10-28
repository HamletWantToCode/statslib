# base optimizer 
# batch gradient descent method

import numpy as np 

class baseOptimizer(object):
    def __init__(self, n_batch, learning_rate, stopError, maxiters):
        self.nb_ = n_batch
        self.lr_ = learning_rate
        self.stopError_ = stopError
        self.maxiters_ = maxiters

    def optimizer(self, alpha, gradient, X, y):
        pass

    def batch_minimize(self, alpha0, function, gradient, X, y, debug=True):
        from .utils import data_spliter
        data = np.c_[X, y]
        loader = data_spliter(data, self.nb_)
        m = 1
        for batch in loader:
            batch_X, batch_y = batch[:,:-1], batch[:, -1]
            if debug and m%3==1:
                print(function(alpha0, batch_X, batch_y))
            alpha = self.optimizer(alpha0, gradient, batch_X, batch_y)
            err = abs(function(alpha0, batch_X, batch_y) - function(alpha, batch_X, batch_y))
            if err < self.stopError_:
                self.alpha_ = alpha
                print('optimization converge after %s epoches and %s inner loops!' %(n_epoch, n_innerloop))
                break
            alpha0 = alpha
            m += 1
        return alpha

    def run(self, alpha0, function, gradient, X, y):
        n_epoch = 1
        while True:
            alpha = self.batch_minimize(alpha0, function, gradient, X, y)
            if n_epoch > self.maxiters_:
                raise ValueError('loops exceed the maximum iteration !')
            alpha0 = alpha
            n_epoch += 1
        return self