# test multi-dim input gp

import numpy as np 
from statslib.main.gauss_process import Gauss_Process_Regressor
from statslib.tools.utils import rbfKernel, rbfKernel_gd, rbfKernel_hess

gamma = 1
kernel = rbfKernel(gamma)
kernel_gd = rbfKernel_gd(gamma)
kernel_hess = rbfKernel_hess(gamma)

def f(x):
    return np.sin(x[0])*np.cos(x[1])

def df(x):
    return np.array([np.cos(x[0])*np.cos(x[1]), -np.sin(x[0])*np.sin(x[1])])

X = np.arange(0.25*np.pi, 1.75*np.pi, 0.25*np.pi)
xx, yy = np.meshgrid(X, X)
xy = np.c_[xx.reshape((-1, 1)), yy.reshape((-1, 1))]
zz = np.array([f(x) for x in xy])
dzz = np.array([df(x) for x in xy])
zz_ = np.r_[zz, dzz.reshape(-1)]

gp = Gauss_Process_Regressor(kernel, 0, kernel_gd, kernel_hess)
gp.fit(xy, zz_[:, np.newaxis])
z_pred, z_pred_err = gp.predict(xy)

import matplotlib.pyplot as plt 
plt.plot(zz, z_pred, 'bo')
plt.plot(zz, zz, 'r')
plt.show()