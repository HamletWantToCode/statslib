# fit boston data with GP

import pickle
import numpy as np 
from statslib.main.gauss_process import Gauss_Process_Regressor
from statslib.tools.utils import rbfKernel, linearKernel

gamma = 1

with open('/Users/hongbinren/Documents/program/statslib/toydataset/boston_data', 'rb') as f:
    data = pickle.load(f)
np.random.shuffle(data)

train_X, train_y = data[:50, :-1], data[:50, -1]
test_X, test_y = data[60:80, :-1], data[60:80, -1]
mean = np.mean(train_y)
train_y -= mean
test_y -= mean

kernel = linearKernel
# kernel = rbfKernel(gamma)
# kernel_gd = rbfKernel_gd(gamma)
# kernel_hess = rbfKernel_hess(gamma)
gp = Gauss_Process_Regressor(kernel, 1e-5)
gp.fit(train_X, train_y[:, np.newaxis])
y_pred, y_pred_err = gp.predict(train_X)

import matplotlib.pyplot as plt 
plt.plot(train_y, train_y, 'r')
plt.plot(train_y, y_pred, 'bo')
plt.show()