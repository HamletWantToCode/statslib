# linear equation solver

import numpy as np 

# direct method
def svd_solver(A, b, cond=None):
    U, S, Vh = np.linalg.svd(A)
    if cond is None:
        cond = np.finfo(np.float64).eps
    rank = len(S[S>cond])
    coefficients = np.divide((U.T[:rank] @ b), S[:rank])
    x = np.zeros(A.shape[1])
    for j, cj in enumerate(coefficients):
        x += cj*Vh[j]
    return x

def cholesky_solver(A, y):
    n = A.shape[1]
    L = np.linalg.cholesky(A)
    b = np.zeros(n)
    x = np.zeros(n)
    for i in range(n):
        b[i] = (y[i] - (L[i] @ b)) / L[i, i]
    for j in range(n, 0, -1):
        x[j-1] = (b[j-1] - (L[:, j-1] @ x)) / L[j-1, j-1]
    return x 

# iteration method
def cg_solver(A, b, x0=None, err=1e-5, maxiter=1000):
    if x0 is None:
        x0 = np.zeros(A.shape[1])
    r0 = b - A @ x0
    p0 = r0
    k = 1
    while True:
        alpha = (r0 @ r0) / (p0 @ (A @ p0))
        x = x0 + alpha*p0
        r1 = r0 - alpha*(A @ p0)
        Err = np.sqrt(r1 @ r1)
        if Err < err:
            print('converge after %s iterations!' %(k))
            x0 = x
            break
        elif k > maxiter:
            print('after %s steps, current error %.3f, not converged !' %(k, Err))
            x0 = x
            break
        beta = (r1 @ r1) / (r0 @ r0)
        p1 = r1 + beta*p0
        r0, p0, x0 = r1, p1, x
        k += 1
    return x0

def RileyGolub_solver(K, y, alpha=None, r0=None, err=1e-5):
    m, n = K.shape
    if alpha is None:
        alpha = np.zeros(n)
    if r0 is None:
        _, S, _ = np.linalg.svd(K)
        lamda_min = S.min()
        r0 = lamda_min*10**(0.5*abs(np.log10(lamda_min)) + 1)
    A = K + r0*np.eye(m)
    k = 1
    Err0 = np.inf
    while True:
        y_ = y + r0*alpha
        alpha_ = cholesky_solver(A, y_)
        diff = alpha_ - alpha
        Err = np.sqrt(diff @ diff)
        if Err < err:
            print('Converged after %s iterations, the error is %.7f' %(k, Err))
            alpha = alpha_
            break
        if Err / Err0 > 0.75:
            r0 = r0 / 2.0
        elif Err / Err0 < 0.25:
            r0 *= 2.0 
        alpha, Err0 = alpha_, Err
        k += 1
    return alpha

if __name__ == '__main__':
    from scipy.linalg import hilbert

    H20 = hilbert(20)
    print(np.linalg.cond(H20))
    X_precise = np.ones(20)
    y = H20 @ X_precise

    X_optim = RileyGolub_solver(H20, y)
    diff = X_optim - X_precise
    print(np.sqrt(diff @ diff))