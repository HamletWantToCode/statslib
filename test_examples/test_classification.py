# classifier test

import pickle
import numpy as np 
from statslib.main.svm import hingeLossSVC
from statslib.main.optimization import GradientDescent
from statslib.tools.utils import linearKernel, classifyAccuracy

with open('test_examples/classification_sample_data', 'rb') as f:
    data = pickle.load(f)
np.random.shuffle(data)

train_X, train_y = data[:100, :-1], data[:100, -1][:, np.newaxis]
test_X, test_y = data[-200:, :-1], data[-200:, -1][:, np.newaxis]
mean = np.mean(train_X, axis=0)
train_X -= mean
test_X -= mean

optimizer = GradientDescent(0.1, 1e-4, 1000, n_batch=1, verbose=1)
model = hingeLossSVC(linearKernel, 1e-5, optimizer)
model.fit(train_X, train_y)
predict_y = model.predict(test_X)

acc = classifyAccuracy(predict_y, test_y)
print(acc)

