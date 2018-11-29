# test multi-task

import numpy as np 
import pickle
from statslib.tools.multitask import Multi_task_Regressor
from statslib.tools.utils import rbfKernel

def regular_func(x):
    return 0

def regular_grad(x):
    return 0

kernel = rbfKernel(0.1)
model = Multi_task_Regressor(kernel, 1e-5, None)
model.regular_func = regular_func
model.regular_grad = regular_grad
full_KM = kernel()