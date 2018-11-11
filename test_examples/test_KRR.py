import pickle
import numpy as np 
from statslib.tools.utils import linearKernel, meanSquareError
from statslib.main.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt  

fname = '/Users/hongbinren/Documents/program/statslib/toydataset/boston_data'
with open(fname, 'rb') as f:
    data = pickle.load(f)
np.random.shuffle(data)
train_X, train_y = data[:200, :-1], data[:200, -1]
test_X, test_y = data[-300:, :-1], data[-300:, -1]

model = KernelRidge(linearKernel, 0)
model.fit(train_X, train_y)
predict_y = model.predict(test_X)

print(meanSquareError(predict_y, test_y))

plt.plot(test_y, predict_y, 'bo')
plt.plot(test_y, test_y, 'r')
plt.show()