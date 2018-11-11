# test module 

import numpy as np 

# optimization
class testOptimize(object):
    def __init__(self, learning_rate, stopError, maxiters):
        self.lr_ = learning_rate
        self.stopError_ = stopError
        self.maxiters_ = maxiters
        
    def run(self, alpha, function, gradient):
        i = 0
        f0 = function(alpha)
        while True:
            alpha = self.optimizer(alpha, gradient)
            f1 = function(alpha)
            print(f1)
            ferr = abs(f1 - f0)
            if ferr < self.stopError_:
                print('optimization converge after %s steps !' %(i))
                break
            elif i > self.maxiters_:
                raise StopIteration('loop exceeds the maximum iterations, fvalue error=%.3f !' %(ferr))
            f0 = f1
            i += 1
        return alpha
    
