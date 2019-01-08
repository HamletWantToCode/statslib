# test gauss process regressor in 1D

import numpy as np 

from statslib.main.gauss_process import GaussProcess 
from statslib.main.utils import rbf_kernel, rbf_kernel_gradient, rbf_kernel_hessan

gamma = 0.1

def kernel(gamma, x, y):
    dx = x - y.T
    return np.exp(-gamma*dx**2)

def kernel_gd(gamma, x, y):
    dx = x - y.T
    return -2*gamma*dx*np.exp(-gamma*dx**2)
    
def kernel_hess(gamma, x, y):
    dx = x - y.T
    return 2*gamma*(1-2*gamma*dx**2)*np.exp(-gamma*dx**2) 

def f(x):
    return x*np.sin(x)

def df(x):
    return np.sin(x) + x*np.cos(x)

X = np.arange(0.25*np.pi, 2*np.pi, 0.5*np.pi)
y = f(X)
dy = df(X)
# dy = np.zeros(len(X))

gp = GaussProcess(gamma=gamma, kernel=rbf_kernel, gradient_on=True, kernel_gd=rbf_kernel_gradient, kernel_hess=rbf_kernel_hessan)
gp.fit(X[:, np.newaxis], y, dy[:, np.newaxis])

Xt = np.linspace(0, 2*np.pi, 50)
yt = f(Xt)
dyt = df(Xt)
y_pred = gp.predict(Xt[:, np.newaxis])
y_var = gp.predict_variance(Xt[:, np.newaxis])
y_gd = gp.predict_gradient(Xt[:, np.newaxis])

import matplotlib.pyplot as plt 
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(X, y, 'ko')
ax1.plot(Xt, yt, 'r')
ax1.plot(Xt, y_pred, 'b', alpha=0.5)
ax1.fill_between(Xt, y_pred-y_var, y_pred+y_var, color='b', alpha=0.5)

ax2.plot(Xt, dyt, 'r')
ax2.plot(Xt, y_gd, 'b--', alpha=0.5)
plt.show()
