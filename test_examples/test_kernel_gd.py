# test kernel gd

import numpy as np 
from statslib.tools.utils import rbfKernel, rbfKernel_gd

np.random.seed(5)

A = np.random.rand(10, 3)
x = np.random.rand(3)[np.newaxis, :]

gamma = 0.1
kernel = rbfKernel(gamma)
kernel_gd = rbfKernel_gd(gamma)

f0 = kernel(x, A)
step = np.random.rand(3)*1e-3
x1 = x + step
f1 = kernel(x1, A)
gd = kernel_gd(x, A)
f_gd = f0 + np.transpose(gd, axes=(0, 2, 1)) @ step
err = abs(f_gd - f1)
print(err)

