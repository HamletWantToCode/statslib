# gradient check
import pickle
import numpy as np 
from statslib.main.svm import *
from statslib.tools.utils import linearKernel, regularizer, regularizer_gradient

def numerical_check(x, f, df, h, *args):
    f0 = f(x, *args)
    gradient = df(x, *args)
    x_ahead = x + h*gradient
    f_ahead = f(x_ahead, *args)
    f_grad = f0 + h*(gradient @ gradient)
    return abs(f_ahead - f_grad)

np.random.seed(5)
fname = '/Users/hongbinren/Documents/program/statslib/toydataset/boston_data'
with open(fname, 'rb') as f:
    data = pickle.load(f)
np.random.shuffle(data)
train_X, train_y = data[:200, :-1], data[:200, -1]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
trans_trainX = scaler.fit_transform(train_X)
# mean = np.mean(train_X, axis=0)
# center_trainX = train_X - mean
# full_KM = np.random.rand(200, 200)
# full_KM = (full_KM + full_KM.T) / 2.0
# train_y = full_KM @ np.ones(200)
# print(np.linalg.cond(full_KM))

full_KM = linearKernel(trans_trainX)
lambda_ = 1e-6
model = EpsilonInsensitiveLossSVR(linearKernel, lambda_, None, 5)
model.regular_func = regularizer(full_KM)
model.regular_grad = regularizer_gradient(full_KM)

alpha = np.random.uniform(0, 10, 200)
h = 1e-3
err = numerical_check(alpha, model.lossFunction, model.lossGradient, h, full_KM, train_y)
print(err)