# optimizer instance

import numpy as np 
from main.baseOptimize import baseOptimizer

class GradientDescent(baseOptimizer):
    def optimizer(self, alpha0, derivative):
        alpha = alpha0 - self.lr_*derivative(alpha0)
        return alpha 

class NAGMethod(baseOptimizer):
    def __init__(self, learning_rate, momentum_param, stopError, maxiters):
        super().__init__(learning_rate, stopError, maxiters)
        self.mp_ = momentum_param
        self.v0_ = 0

    def optimizer(self, alpha0, derivative):
        alphaHEAD = alpha0 + self.mp_*self.v0_
        v = self.mp_*self.v0_ - self.lr_*derivative(alphaHEAD)
        alpha = alpha0 + v
        self.v0_ = v
        return alpha

class RMSprop(baseOptimizer):
    def __init__(self, learning_rate, rms_param, stopError, maxiters):
        super().__init__(learning_rate, stopError, maxiters)
        self.rmsParam_ = rms_param
        self.s0_ = 0
        
    def optimizer(self, alpha0, derivative):
        s = self.rmsParam_*self.s0_ + (1 - self.rmsParam_)*(derivative(alpha0))**2
        alpha = alpha0 - self.lr_*(1.0 / np.sqrt(s + 1e-8))*derivative(alpha0)
        self.s0_ = s
        return alpha

class Adam(baseOptimizer):
    def __init__(self, learning_rate, rms_param, momentum_param, stopError, maxiters):
        super().__init__(learning_rate, stopError, maxiters)
        self.rmsParam_ = rms_param
        self.mp_ = momentum_param
        self.v0_ = 0
        self.s0_ = 0
        self.t_ = 0

    def optimizer(self, alpha0, derivative):
        self.t_ += 1
        v = self.mp_*self.v0_ + (1 - self.mp_)*derivative(alpha0)
        s = self.rmsParam_*self.s0_ + (1 - self.rmsParam_)*(derivative(alpha0))**2
        v_corr = v / (1 - self.mp_**self.t_)
        s_corr = s / (1 - self.rmsParam_**self.t_)
        alpha = alpha0 - self.lr_*(1.0 / np.sqrt(s_corr + 1e-8))*v_corr
        self.v0_ = v
        self.s0_ = s
        return alpha

if __name__ == '__main__':
    import numpy as np 

    f = lambda x: (1-x[0])**2 + 100*(x[1]-x[0])**2
    df = lambda x: np.array([-2*(1-x[0])-200*(x[1]-x[0]), 200*(x[1]-x[0])])
    xinit = np.array([-0.6, 0.4])
    # optimizer = GradientDescent(0.005, 1e-4, 1000)
    optimizer = Adam(0.5, 0.9, 0.9, 1e-4, 10000)
    optimizer.run(xinit, f, df)
    print(optimizer.alpha_)