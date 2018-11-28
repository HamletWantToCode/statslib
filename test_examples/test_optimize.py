import pickle
import numpy as np 
from statslib.main.optimization import GradientDescent
from statslib.main.svm import hingeLossSVC
from statslib.tools.utils import linearKernel, classifyAccuracy

with open('toydataset/breast_cancer_data', 'rb') as f:
    data = pickle.load(f)
np.random.shuffle(data)

train_X, train_y = data[:200, :-1], data[:200, -1][:, np.newaxis]
mean = np.mean(train_X, axis=0)
train_X -= mean
var = np.var(train_X, axis=0)
train_X /= np.sqrt(var)

test_X, test_y = data[200:, :-1], data[200:, -1][:, np.newaxis]
test_X -= mean
test_X /= np.sqrt(var)

optimizer = GradientDescent(0.01, 1e-4, 1000, n_batch=20, verbose=1)
model = hingeLossSVC(linearKernel, 1e-5, optimizer)
model.fit(train_X, train_y)
predict_y = model.predict(test_X)

acc = classifyAccuracy(predict_y, test_y)

import matplotlib.pyplot as plt 

fvals = optimizer.fvals_
N = len(fvals)
plt.plot(range(N), fvals)
plt.text(N/2, (np.amax(fvals)+np.amin(fvals))/2, 'accuracy=%.3f' %(acc))
plt.xlabel('epoch')
plt.ylabel('y')
plt.show()

