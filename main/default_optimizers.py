# optimizer instance

from main.baseOptimize import baseOptimizer

class GradientDescent(baseOptimizer):
    def optimizer(self, alpha0, derivative):
        alpha = alpha0 - self.lr_*derivative(alpha0)
        return alpha 

class Momentum(baseOptimizer):
    def __init__(self, learning_rate, momentum_param, initVelocity, stopError, maxiters):
        super().__init__(learning_rate, stopError, maxiters)
        self.mp_ = momentum_param
        self.v0_ = initVelocity

    def optimizer(self, alpha0, derivative):
        v = self.mp_*self.v0_ - self.lr_*derivative(alpha0)
        alpha = alpha0 + v
        self.v0_ = v
        return alpha

if __name__ == '__main__':
    import numpy as np 

    f = lambda x: (1-x[0])**2 + 100*(x[1]-x[0])**2
    df = lambda x: np.array([-2*(1-x[0])-200*(x[1]-x[0]), 200*(x[1]-x[0])])
    xinit = np.array([-0.6, 0.4])
    optimizer = GradientDescent(1e-3, 1e-5, 1e4)
    optimizer.run(xinit, f, df)
    print(optimizer.alpha_)