import pickle
import numpy as np 

from tools.classifier import *
from main.default_kernels import *
from main.default_optimizers import *
from main.metric import *

with open('/Users/hongbinren/Documents/program/svm/toydataset/iris_data', 'rb') as f:
    Data = pickle.load(f)

Target = np.zeros_like(Data[:, -1])
Target[Data[:, -1]==1] = 1
Target[Data[:, -1]!=1] = -1

trainfeature, traintarget = Data[:80, :-1], Target[:80]
testfeature, testtarget = Data[80:, :-1], Target[80:]

optimizer = NAGMethod(0.1, 0.80, 1e-3, 10000)
model = hingeLossSVC(rbfKernel, 1e-4, optimizer)
model.fit(trainfeature, traintarget)
predictlabels = model.predict(trainfeature)
# print(predictlabels)
# print(traintarget)

precision = classifyAccuracy(predictlabels, traintarget)
print(precision)