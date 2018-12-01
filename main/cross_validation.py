# cross-validation

import numpy as np 

class Cross_validation(object):
    def __init__(self, data_gen, solver, measure):
        self.id = data_gen
        self.solver = solver
        self.measure = measure

    def run(self, X, y):
        performance = 0
        n = 0
        for (test_id, valid_id) in self.id:
            trainX, trainy = X[valid_id], y[valid_id]
            testX, testy = X[test_id], y[test_id]
            self.solver.fit(trainX, trainy)
            predicty = self.solver.predict(testX)
            score = self.measure(np.squeeze(predicty), np.squeeze(testy))
            performance += score**2
            n += 1
        return np.sqrt(performance*1.0/n)
        
