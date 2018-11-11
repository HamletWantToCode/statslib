import pickle
import numpy as np 
import matplotlib.pyplot as plt  

np.random.seed(5)
fname = '/Users/hongbinren/Documents/program/statslib/toydataset/boston_data'
with open(fname, 'rb') as f:
    data = pickle.load(f)
np.random.shuffle(data)
train_X, train_y = data[:200, :-1], data[:200, -1]
test_X, test_y = data[-300:, :-1], data[-300:, -1]
mean = np.mean(train_X, axis=0)
center_trainX = train_X - mean
center_testX = test_X - mean

# from statslib.main.svm import EpsilonInsensitiveLossSVR
# from statslib.tools.utils import linearKernel, meanSquareError
# from statslib.main.optimization import GradientDescent
# optimizer = GradientDescent(1e-9, 1e-3, 500, n_batch=200)
# lambda_ = 1e-4
# epsilon = 10
# model = EpsilonInsensitiveLossSVR(linearKernel, lambda_, optimizer, epsilon)
# model.fit(center_trainX, train_y)
# predict_y = model.predict(center_testX)
# err = meanSquareError(predict_y, test_y)

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
model = SGDRegressor(loss='epsilon_insensitive', epsilon=10, alpha=1e-4, learning_rate='constant', eta0=1e-3, max_iter=20, shuffle=False, verbose=1)
model.fit(center_trainX, train_y)
predict_y = model.predict(center_testX)
err = mean_squared_error(predict_y, test_y)

print(err)
plt.plot(test_y, predict_y, 'bo')
plt.plot(test_y, test_y, 'r')
plt.show()
