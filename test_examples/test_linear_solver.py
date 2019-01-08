# test solver performance in solving ill-conditioned
# linear equations

import numpy as np 
from scipy.linalg import hilbert

from statslib.main.utils import svd_solver, cholesky_solver, iterative_solver

R = np.random.RandomState(584392)
A = hilbert(20)
x = R.uniform(-10, 10, 20)
b = A @ x
k = 1e-10

x_svd = svd_solver(A, b, k, cond=1e-8)
x_ch = cholesky_solver(A, b, k)
x_iter = iterative_solver(A, b, k, max_iters=5000, std_err=1e-8)

err_svd = np.sqrt(np.mean((x_svd - x)**2))
err_ch = np.sqrt(np.mean((x_ch - x)**2))
err_iter = np.sqrt(np.mean((x_iter - x)**2))

print(err_svd, err_ch, err_iter)
