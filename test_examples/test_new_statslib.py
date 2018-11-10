from statslib.main.optimization import GradientDescent
from statslib.main.svm import hingeLossSVC
from statslib.tools.utils import linearKernel, classifyAccuracy
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import SGDClassifier
import pickle
import numpy as np 

fname = '/Users/hongbinren/Documents/program/statslib/toydataset/Breast_cancer_data'
with open(fname, 'rb') as f:
    data = pickle.load(f)
np.random.shuffle(data)

scaler = StandardScaler()
train_X, train_y = data[:200, :-1], data[:200, -1]
test_X, test_y = data[-300:, :-1], data[-300:, -1]
mean = np.mean(train_X, axis=0)
# normal_trainX = train_X - mean
# normal_testX = test_X - mean
normal_trainX = scaler.fit_transform(train_X)
normal_testX = scaler.transform(test_X)

lambda_ = 1e-4
optimizer = GradientDescent(0.01, 1e-3, 500, n_batch=10)
model = hingeLossSVC(linearKernel, lambda_, optimizer)
model.fit(normal_trainX, train_y)
predict_y = model.predict(normal_testX)

# model = SGDClassifier(alpha=1e-4, learning_rate='constant', eta0=0.01, max_iter=500, fit_intercept=False, verbose=1)
# model.fit(normal_trainX, train_y)
# predict_y = model.predict(normal_testX)

from sklearn.metrics import accuracy_score
acc = accuracy_score(predict_y, test_y)
print(acc)