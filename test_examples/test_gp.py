# test gauss process regression

import numpy as np 
from statslib.main.gauss_process import Gauss_Process_Regressor
from statslib.tools.utils import rbfKernel
import matplotlib.pyplot as plt 

def f(x):
    return np.sin(3*x)

train_X = np.arange(0.5*np.pi, 2*np.pi, 0.5*np.pi)
train_y = f(train_X)

gamma = 1
kernel = rbfKernel(gamma)
sigma = 0.01
gp = Gauss_Process_Regressor(kernel, sigma)
gp.fit(train_X[:, np.newaxis], train_y[:, np.newaxis])

X = np.linspace(0, 2*np.pi, 50)
y = f(X)
predict_y, predict_error = gp.predict(X[:, np.newaxis])

plt.plot(X, y, 'r')
plt.plot(train_X, train_y, 'ko')
plt.plot(X, predict_y, 'b', alpha=0.5)
plt.plot(X, predict_y-predict_error, 'b--', alpha=0.5)
plt.plot(X, predict_y+predict_error, 'b--', alpha=0.5)
plt.show()




