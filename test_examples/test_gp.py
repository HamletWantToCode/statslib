# test general covariance

import numpy as np 
from statslib.main.gauss_process import Gauss_Process_Regressor
from MLEK.tools.kernels import se_kernel, se_kernel_gd, se_kernel_hess 

gamma = np.array([1])

# def kernel(x, y=None):
#     if y is None:
#         y = x
#     dx = x - y.T
#     return np.exp(-gamma*dx**2)

# def kernel_gd(x, y=None):
#     if y is None:
#         y = x
#     dx = x - y.T
#     return -2*gamma*dx*np.exp(-gamma*dx**2)
    
# def kernel_hess(x, y=None):
#     if y is None:
#         y = x
#     dx = x - y.T
#     return 2*gamma*(1-2*gamma*dx**2)*np.exp(-gamma*dx**2) 

def f(x):
    return x*np.sin(x)

def df(x):
    return np.sin(x) + x*np.cos(x)

X = np.arange(0.25*np.pi, 2*np.pi, 0.5*np.pi)
y = f(X)
dy = df(X)
y_ = np.r_[y, dy]

kernel = se_kernel(gamma)
kernel_gd = se_kernel_gd(gamma)
kernel_hess = se_kernel_hess(gamma)

gp = Gauss_Process_Regressor(kernel, 1e-5, 1e-5, kernel_gd, kernel_hess)
gp.fit(X[:, np.newaxis], y_[:, np.newaxis])

Xt = np.linspace(0, 2*np.pi, 50)
yt = f(Xt)
y_pred, y_error = gp.predict(Xt[:, np.newaxis])
# dK = kernel_gd(Xt[:, np.newaxis], X[:, np.newaxis])
# predict_dy = dK @ gp.coef_
# ddK = kernel(X[:, np.newaxis]) 
# ddK_predict = kernel_hess(Xt[:, np.newaxis], Xt[:, np.newaxis])
# ddK_corr = kernel_gd(Xt[:, np.newaxis], X[:, np.newaxis])
# predict_dy_error = np.diag(ddK_predict - ddK_corr @ np.linalg.pinv(ddK) @ ddK_corr.T)

import matplotlib.pyplot as plt 
plt.plot(X, y, 'ko')
plt.plot(Xt, yt, 'r')
plt.plot(Xt, y_pred, 'b', alpha=0.5)
plt.fill_between(Xt, y_pred-y_error, y_pred+y_error, color='b', alpha=0.5)
plt.show()
