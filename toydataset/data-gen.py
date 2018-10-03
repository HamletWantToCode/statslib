# toy dataset

import numpy as np 
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_boston
import pickle 

# Gaussian data
numOfDataInSet1 = 500
numOfDataInSet2 = 500

dataSet1X, dataSet1Y = np.random.multivariate_normal([3, 3], [[2, 0], [0, 4]], numOfDataInSet1).T
dataLabel1 = np.ones(numOfDataInSet1)

dataSet2X, dataSet2Y = np.random.multivariate_normal([-3, -3], [[2, 0], [0, 4]], numOfDataInSet2).T
dataLabel2 = -1*np.ones(numOfDataInSet2)

data1 = np.c_[dataSet1X.reshape(-1, 1), dataSet1Y.reshape(-1, 1), dataLabel1.reshape(-1, 1)]
data2 = np.c_[dataSet2X.reshape(-1, 1), dataSet2Y.reshape(-1, 1), dataLabel2.reshape(-1, 1)]
Gaussian_data = np.r_[data1, data2]

np.random.shuffle(Gaussian_data)

with open('Gaussian_data', 'wb') as f:
    pickle.dump(Gaussian_data, f)

# breast cancer data
Feature, Target = load_breast_cancer(return_X_y=True)
Target[Target==0] = -1
Breast_cancer_data = np.c_[Feature, Target]
np.random.shuffle(Breast_cancer_data)

with open('Breast_cancer_data', 'wb') as f:
    pickle.dump(Breast_cancer_data, f)

# iris data
Feature, Target = load_iris(return_X_y=True)
irisData = np.c_[Feature, Target]
np.random.shuffle(irisData)

with open('iris_data', 'wb') as f:
    pickle.dump(irisData, f)
    
# wine data
Feature, Target = load_wine(return_X_y=True)
wineData = np.c_[Feature, Target]
np.random.shuffle(wineData)

with open('wine_data', 'wb') as f:
    pickle.dump(wineData, f)

# boston data
Feature, Target = load_boston(return_X_y=True)
bostonData = np.c_[Feature, Target]
np.random.shuffle(bostonData)

with open('boston_data', 'wb') as f:
    pickle.dump(bostonData, f)