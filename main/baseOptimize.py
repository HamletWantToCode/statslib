# base optimizer 
# batch gradient descent method

class baseOptimizer(object):
    def __init__(self, learning_rate, stopError, maxiters):
        self.lr_ = learning_rate
        self.stopError_ = stopError
        self.maxiters_ = maxiters

    def optimizer(self, alpha, derivative):
        pass

    def run(self, alpha0, function, derivative, debug=False):
        i = 0
        while True:
            if debug and i%5==0:
                print(function(alpha0))
            i += 1
            alpha = self.optimizer(alpha0, derivative)
            if abs(function(alpha) - function(alpha0)) < self.stopError_:
                self.alpha_ = alpha
                print('optimization converge after %s steps!' %(i))
                break
            elif i > self.maxiters_:
                raise ValueError('loops exceed the maximum iteration !')
            alpha0 = alpha
        return self