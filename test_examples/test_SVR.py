import pickle
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler

np.random.seed(5)
fname = '/Users/hongbinren/Documents/program/statslib/toydataset/boston_data'
with open(fname, 'rb') as f:
    data = pickle.load(f)
np.random.shuffle(data)
scaler = StandardScaler()
train_X, train_y = data[:200, :-1], data[:200, -1]
test_X, test_y = data[-300:, :-1], data[-300:, -1]
mean_y = np.mean(train_y)
centered_trainy = train_y - mean_y
normal_trainX = scaler.fit_transform(train_X)
normal_testX = scaler.transform(test_X)
centered_testy = test_y - mean_y

from statslib.main.svm import EpsilonInsensitiveLossSVR
from statslib.tools.utils import linearKernel, meanSquareError
from statslib.main.optimization import * 
optimizer = GradientDescent(1e-3, 1e-3, 500, n_batch=20)
# optimizer = NesterovGD(1e-3, 1e-3, 500, 0.9, n_batch=1)
lambda_ = 1e-5
epsilon = 5
model = EpsilonInsensitiveLossSVR(linearKernel, lambda_, optimizer, epsilon)
model.fit(normal_trainX, centered_trainy)
predict_y = model.predict(normal_testX)
err = meanSquareError(predict_y, centered_testy)

# from sklearn.linear_model import SGDRegressor
# from sklearn.metrics import mean_squared_error
# model = SGDRegressor(loss='epsilon_insensitive', epsilon=10, alpha=1e-4, learning_rate='constant', eta0=1e-3, max_iter=20, shuffle=False, fit_intercept=False, verbose=1)
# model.fit(center_trainX, train_y)
# predict_y = model.predict(center_testX)
# err = mean_squared_error(predict_y, test_y)

# from statslib.tools.utils import svd_solver
# A = center_trainX.T 
# b = model.coef_
# x = svd_solver(A, b)

print(err)
plt.plot(centered_testy, predict_y, 'bo')
plt.plot(centered_testy, centered_testy, 'r')
plt.show()

