# test kernel gd

import numpy as np 
from statslib.tools.utils import rbfKernel, rbfKernel_gd, rbfKernel_hess

np.random.seed(5)

A = np.random.rand(10, 3)
B = np.random.rand(5, 3)

gamma = 0.1
kernel = rbfKernel(gamma)
kernel_gd = rbfKernel_gd(gamma)
kernel_hess = rbfKernel_hess(gamma)

f0 = kernel(B, A)
step = np.ones(3)*1e-3
B1 = B + step
f1 = kernel(B1, A)
gd = (kernel_gd(B, A)).reshape((5, 3, 10))
f_gd = f0 + np.sum(gd*step[np.newaxis, :, np.newaxis], axis=1)
err = abs(f_gd - f1)
# print(err)

A1 = A + step
f1_ = kernel(B1, A1)
hess = (kernel_hess(B, A)).reshape((5, 3, 3, 10))
f_hess = f0 + 2*np.sum(gd*step[np.newaxis, :, np.newaxis], axis=1) #+ 0.5*np.sum(hess*step[np.newaxis, :, np.newaxis, np.newaxis]*step[np.newaxis, np.newaxis, :, np.newaxis], axis=(1, 2)) 
err_ = abs(f_hess - f1_)
print(err_)
