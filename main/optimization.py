# optimizer instance

import numpy as np 
from .base import BaseOptimize
# from ..test_examples.opt_test_util import testOptimize

class GradientDescent(BaseOptimize):
    def __init__(self, learning_rate, stopError, maxiters, **kwargs):
        super().__init__(learning_rate, stopError, maxiters, **kwargs)

    def optimizer(self, alpha, gradient, *args):
        return alpha - self.lr_*gradient(alpha, *args)


class NesterovGD(BaseOptimize):
    def __init__(self, learning_rate, stopError, maxiters, momentum_param, **kwargs):
        super().__init__(learning_rate, stopError, maxiters, **kwargs)
        self.mp_ = momentum_param
        self.v0_ = 0

    def optimizer(self, alpha, gradient, *args):
        v = self.mp_*self.v0_ - self.lr_*gradient(alpha, *args)
        alpha = alpha - self.mp_*self.v0_ + (1 + self.mp_)*v
        self.v0_ = v
        return alpha

# class RMSprop(BaseOptimizer):
#     def __init__(self, learning_rate, stopError, maxiters, rms_param, **kwargs):
#         super().__init__(learning_rate, stopError, maxiters, **kwargs)
#         self.rmsParam_ = rms_param
#         self.s0_ = 0
        
#     def optimizer(self, alpha0, gradient, *args):
#         s = self.rmsParam_*self.s0_ + (1 - self.rmsParam_)*(gradient(alpha0, *args))**2
#         alpha = alpha0 - self.lr_*(1.0 / np.sqrt(s + 1e-8))*gradient(alpha0, *args)
#         self.s0_ = s
#         return alpha

# class Adam(BaseOptimizer):
#     def __init__(self, learning_rate, stopError, maxiters, rms_param, momentum_param, **kwargs):
#         super().__init__(learning_rate, stopError, maxiters, **kwargs)
#         self.rmsParam_ = rms_param
#         self.mp_ = momentum_param
#         self.v0_ = 0
#         self.s0_ = 0
#         self.t_ = 0

#     def optimizer(self, alpha0, gradient, *args):
#         self.t_ += 1
#         v = self.mp_*self.v0_ + (1 - self.mp_)*gradient(alpha0, *args)
#         s = self.rmsParam_*self.s0_ + (1 - self.rmsParam_)*(gradient(alpha0, *args))**2
#         v_corr = v / (1 - self.mp_**self.t_)
#         s_corr = s / (1 - self.rmsParam_**self.t_)
#         alpha = alpha0 - self.lr_*(1.0 / np.sqrt(s_corr + 1e-8))*v_corr
#         self.v0_ = v
#         self.s0_ = s
#         return alpha