# test multi-dim input gp

import numpy as np 
from statslib.main.gauss_process import Gauss_Process_Regressor
# from statslib.tools.utils import rbfKernel, rbfKernel_gd, rbfKernel_hess
from MLEK.tools.kernels import se_kernel, se_kernel_gd, se_kernel_hess

gamma = np.array([1.0, 1.0])
kernel = se_kernel(gamma)
kernel_gd = se_kernel_gd(gamma)
kernel_hess = se_kernel_hess(gamma)

def f(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

def df(x):
    return np.array([-2*(1-x[0])-400*(x[1]-x[0]**2)*x[0], 200*(x[1]-x[0]**2)])

xy = np.random.uniform(-0.6, 1.0, size=(5, 2))
zz = np.array([f(x) for x in xy])
dzz = np.array([df(x) for x in xy])
zz_ = np.r_[zz, dzz.reshape(-1)]

# Xt = np.linspace(0, 2*np.pi, 10)
# xx_t, yy_t = np.meshgrid(Xt, Xt)
# xy_t = np.c_[xx_t.reshape((-1, 1)), yy_t.reshape((-1, 1))]
# zz_t = np.array([f(x) for x in xy_t])

gp = Gauss_Process_Regressor(kernel, 1e-3, kernel_gd, kernel_hess)
gp.fit(xy, zz_[:, np.newaxis])
z_pred, z_pred_err = gp.predict(xy)

Kgd = kernel_gd(xy)
Khess = kernel_hess(xy)
K_star = np.c_[Kgd, Khess]
predict_dy = (K_star @ gp.coef_).reshape((5, 2))

err = np.sum((dzz.reshape((5, 2)) - predict_dy)**2, axis=1)
print(err)
# import matplotlib.pyplot as plt 
# plt.plot(zz, z_pred, 'bo')
# plt.plot(zz, zz, 'r')
# plt.show()