import pickle
import numpy as np 
from statslib.tools.utils import euclidean_distance

fname = '/Users/hongbinren/Documents/program/statslib/toydataset/Breast_cancer_data'
with open(fname, 'rb') as f:
    data = pickle.load(f)
feature = data[:, :-1]

D = euclidean_distance(feature, feature)