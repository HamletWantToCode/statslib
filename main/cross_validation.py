# cross-validation

import numpy as np 

class Cross_validation(object):
    def __init__(self, data_gen, solver, kernel_gd):
        self.id = data_gen
        self.solver = solver
        self.kernel_gd = kernel_gd

    def run(self, X, y, dy):
        performance = 0
        n = 0
        for (test_id, valid_id) in self.id:
            train_X, train_y = X[valid_id], y[valid_id]
            test_X, test_y, test_dy = X[test_id], y[test_id], dy[test_id]
            self.solver.fit(train_X, train_y)
            predict_y = self.solver.predict(test_X)
            K_gd = self.kernel_gd(test_X, train_X)
            predict_dy = 501*(K_gd @ self.solver.coef_).reshape(test_dy.shape)
            score = np.mean((predict_y - test_y)**2) + np.mean(np.mean((predict_dy - test_dy)**2, axis=1))
            performance += score**2
            n += 1
        return np.sqrt(performance*1.0/n)
        
