# test condition number of K

from statslib.main.gauss_process import cov_setup
from statslib.tools.utils import rbfKernel, rbfKernel_gd, rbfKernel_hess
import numpy as np 

X = np.linspace(0, 20, 100)
delta = np.linspace(1e-5, 0.2, 20)
cond = [[], []]
for d in delta:
    scaler = np.sqrt(1./d)
    kernel = rbfKernel(1)
    kernel_gd = rbfKernel_gd(1)
    kernel_hess = rbfKernel_hess(1)
    K_gd = cov_setup(X[:, np.newaxis]*scaler, kernel, kernel_gd, kernel_hess)
    K = cov_setup(X[:, np.newaxis]*scaler, kernel)
    cond[0].append(np.linalg.cond(K))
    cond[1].append(np.linalg.cond(K_gd))

import matplotlib.pyplot as plt 
plt.plot(delta, cond[0], 'b')
plt.plot(delta, cond[1], 'r')
plt.semilogy()
plt.show()