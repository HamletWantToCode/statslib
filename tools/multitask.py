# multi-task learning

import numpy as np 
from statslib.main.base import BaseRegressor
from statslib.main.base import BaseOptimize
from statslib.main.optimization import GradientDescent
from statslib.tools.utils import regularizer, regularizer_gradient

def special_load_data(full_KM, X, y, n_batch=1):
    N = data.shape[0]
    n_elements = N//n_batch
    for i in range(n_batch):
        sub_KM, sub_X, sub_y, sub_dy = full_KM[i*n_elements:(i+1)*n_elements], X[i*n_elements:(i+1)*n_elements], y[i*n_elements:(i+1)*n_elements, 0][:, np.newaxis], y[i*n_elements:(i+1)*n_elements, 1:]
        yield (sub_KM, sub_X, sub_y, sub_dy)

class Multi_task_Regressor(BaseRegressor):
    def __init__(self, kernel, Lambda, optimizer):
        super().__init__(kernel, Lambda, optimizer)

    def lossFunction(self, alpha, sub_KM, sub_X, sub_y, sub_dy, gamma):
        n_samples = sub_KM.shape[0]
        Dy = np.squeeze(sub_y) - (sub_KM @ alpha)
        loss_on_function = 0.5*np.sum(Dy**2)
        dX = sub_X[:, np.newaxis, :] - self.X_fit_
        dK = -2*gamma*dX*sub_KM[:, :, np.newaxis]
        compute_dy = np.sum(alpha[np.newaxis, :, np.newaxis]*dK, axis=1)
        loss_on_gradient = 0.5*np.sum((compute_dy - sub_dy)*((compute_dy - sub_dy).conj()))
        regular_term = self.regular_func(alpha)
        return (1.0/n_samples)*(loss_on_function + loss_on_gradient) + 0.5*self.Lambda_*regular_term

    def lossGradient(self, alpha, sub_KM, sub_X, sub_y, sub_dy, gamma):
        n_samples = sub_KM.shape[0]
        Dy = np.squeeze(sub_y) - (sub_KM @ alpha)
        derivative_on_function = Dy @ sub_KM
        dX = sub_X[:, np.newaxis, :] - self.X_fit_
        dK = -2*gamma*dX*sub_KM[:, :, np.newaxis]
        compute_dy = np.sum(alpha[np.newaxis, :, np.newaxis]*dK, axis=1)
        D_dy_c = (compute_dy - sub_dy).conj()
        derivative_on_gradient = np.sum(np.sum(D_dy_c[:, np.newaxis, :]*dK), axis=0), axis=1)
        regular_grad_term = self.regular_grad(alpha)
        return (1.0/n_samples)*(derivative_on_function + derivative_on_gradient) + self.Lambda_*regular_grad_term

    def fit(self, X, y, gamma):
        n_sample = X.shape[0]
        self.X_fit_ = X
        full_KM = self.kernel(X)
        self.regular_func = regularizer(full_KM)
        self.regular_grad = regularizer_gradient(full_KM)
        alpha0 = np.random.uniform(-1, 1, n_sample)*1e-2
        alpha = self.optimizer.run(alpha0, self.lossFunction, self.lossGradient, full_KM, X, y, gamma)
        self.coef_ = alpha
        return self

class Special_optimizer(object):
    def run(self, alpha, function, gradient, full_KM, X, y, gamma):
            f0 = function(alpha, full_KM, X, y[:, 0][:, np.newaxis], y[:, 1:], gamma)
            n_epoch = 0
            while True:
                loader = special_load_data(full_KM, X, y, self.nb_)
                for i, (sub_KM, sub_X, sub_y, sub_dy) in enumerate(loader):
                    alpha = self.optimizer(alpha, gradient, sub_KM, sub_X, sub_y, sub_dy, gamma)
                    if self.verbose_:
                        f_update = function(alpha, full_KM, X, y, dy, gamma)
                        self.fvals_.append(f_update)
                f1 = function(alpha, full_KM, X, y, dy, gamma)
                ferr = abs(f1-f0)
                if ferr < self.stopError_:
                    print('optimization converges after %d epoches and %d batch iterations!' %(n_epoch, i+1))
                    break
                f0 = f1
                if n_epoch > self.maxiters_:
                    raise StopIteration('loop exceeds the maximum iterations !')
                n_epoch += 1
                if n_epoch > 50:
                    self.lr_ *= 0.99
            return alpha

class Special_SGD(Special_optimizer, GradientDescent):
    def __init__(self, learning_rate, stopError, maxiters, **kwargs):
        super().__init__(learning_rate, stopError, maxiters, **kwargs)
