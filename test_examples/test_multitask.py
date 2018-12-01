# test multi-task

import numpy as np 
import pickle
from statslib.tools.multitask import Multi_task_Regressor
from statslib.tools.utils import rbfKernel

def regular_func(x):
    return 0

def regular_grad(x):
    return 0

fname = '/home/hongbin/Documents/project/MLEK/data_file/quantum'
with open(fname, 'rb') as f:
    data = pickle.load(f)
fname1 = '/home/hongbin/Documents/project/MLEK/data_file/potential'
with open(fname1, 'rb') as f1:
    potential = pickle.load(f1)
dy = np.zeros_like(potential[:100, 1:], dtype=np.complex64)
dy[:100, 0] = potential[:100, 0]
dy -= potential[:100, 1:]
X = data[:100, 1:]
y = data[:100, 0][:, np.newaxis].real 

gamma = 0.1
kernel = rbfKernel(gamma)
model = Multi_task_Regressor(kernel, 0, None)
model.regular_func = regular_func
model.regular_grad = regular_grad
model.X_fit_ = X
full_KM = kernel(X)

def numerical_check(x, f, df, h, *args):
    n = len(x)
    f0 = f(x, *args)
    gradient = df(x, *args)
    Err = []
    for i in range(n):
        step = np.zeros(n)
        step[i] = h
        x_ahead = x + step
        f_ahead = f(x_ahead, *args)
        f_grad = f0 + gradient @ step
        err = abs(f_ahead - f_grad)
        Err.append(err)
    return Err

alpha = np.random.uniform(0, 10, len(X))
model.lossFunction(alpha, full_KM, X, y, dy, gamma)
err = numerical_check(alpha, model.lossFunction, model.lossGradient, 1e-3, full_KM, X, y, dy, gamma)
print(err)