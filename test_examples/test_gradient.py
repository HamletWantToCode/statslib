# gradient check
import pickle
import numpy as np 
from statslib.main.svm import *
from statslib.tools.utils import linearKernel

def regular_func(x):
    return 0

def regular_grad(x):
    return 0

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

np.random.seed(5)
fname = 'test_examples/sample_data'
with open(fname, 'rb') as f:
    data = pickle.load(f)
np.random.shuffle(data)
train_X, train_y = data[:200, :-1], data[:200, -1][:, np.newaxis]
mean = np.mean(train_X, axis=0)
train_X -= mean 

full_KM = linearKernel(train_X)
lambda_ = 0
model = hingeLossSVC(linearKernel, lambda_, None)
model.regular_func = regular_func
model.regular_grad = regular_grad

alpha = np.random.uniform(0, 10, 200)
h = 0.1
err = numerical_check(alpha, model.lossFunction, model.lossGradient, h, full_KM, train_y)
print(err)