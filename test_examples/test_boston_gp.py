# fit boston data with GP

import pickle
import numpy as np 
from statslib.main.gauss_process import Gauss_Process_Regressor
from statslib.tools.utils import rbfKernel
from MLEK.tools.kernels import *

gamma = 0.001

with open('/Users/hongbinren/Documents/program/statslib/toydataset/boston_data', 'rb') as f:
    data = pickle.load(f)
np.random.shuffle(data)

train_X, train_y = data[:300, :-1], data[:300, -1]
test_X, test_y = data[350:400, :-1], data[350:400, -1]
# train_dy = np.random.uniform(-10, 10, size=(50, 13))
mean = np.mean(train_y)
train_y -= mean
# train_y_ = np.r_[train_y, train_dy.reshape(-1)]
test_y -= mean

kernel = rbfKernel(gamma)
# kernel = se_kernel(gamma)
# kernel_gd = se_kernel_gd(gamma)
# kernel_hess = se_kernel_hess(gamma)
gp = Gauss_Process_Regressor(kernel, 1e-2)
gp.fit(train_X, train_y[:, np.newaxis])
y_pred, y_pred_err = gp.predict(test_X)

import matplotlib.pyplot as plt 
plt.plot(test_y, test_y, 'r')
plt.plot(test_y, y_pred, 'bo')
plt.show()