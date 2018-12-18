# cross-validation

import numpy as np 

class Cross_validation(object):
    def __init__(self, data_gen, solver):
        self.id = data_gen
        self.solver = solver

    def run(self, X, y, dy):
        performance = 0
        n = 0
        for (test_id, valid_id) in self.id:
            train_X, train_y, train_dy = X[valid_id], y[valid_id], None
            test_X, test_y, test_dy = X[test_id], y[test_id], dy[test_id]
            self.solver.fit(train_X, train_y, train_dy)
            predict_y, predict_dy = self.solver.predict(test_X)
            project_dy = test_dy @ self.solver.tr_mat_
            score = np.mean((predict_y - test_y)**2) + np.mean(np.mean((predict_dy - project_dy)**2, axis=1))
            performance += score**2
            n += 1
        return np.sqrt(performance*1.0/n)
        
