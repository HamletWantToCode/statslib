# from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np 

np.random.seed(5)
fname = '/Users/hongbinren/Documents/program/statslib/toydataset/Breast_cancer_data'
with open(fname, 'rb') as f:
    data = pickle.load(f)
np.random.shuffle(data)

# scaler = StandardScaler()
train_X, train_y = data[:200, :-1], data[:200, -1]
test_X, test_y = data[-300:, :-1], data[-300:, -1]
mean = np.mean(train_X, axis=0)
normal_trainX = train_X - mean
normal_testX = test_X - mean
# normal_trainX = scaler.fit_transform(train_X)
# normal_testX = scaler.transform(test_X)

from statslib.main.optimization import * 
from statslib.main.svm import hingeLossSVC
from statslib.tools.utils import linearKernel
lambda_ = 0
optimizer = GradientDescent(1e-8, 1e-3, 10, n_batch=200, verbose=1)
# optimizer = NesterovGD(1e-8, 1e-3, 500, 0.2, n_batch=200)
model = hingeLossSVC(linearKernel, lambda_, optimizer)
model.fit(normal_trainX, train_y)
predict_y = model.predict(normal_testX)

# from sklearn.linear_model import SGDClassifier
# model = SGDClassifier(alpha=1e-4, learning_rate='constant', eta0=0.01, max_iter=500, fit_intercept=False)
# model.fit(normal_trainX, train_y)
# predict_y = model.predict(normal_testX)

# from statslib.tools.utils import svd_solver
# b = model.coef_[0]
# A = normal_trainX.T 
# x = svd_solver(A, b)

from sklearn.metrics import accuracy_score
acc = accuracy_score(predict_y, test_y)
print(acc)