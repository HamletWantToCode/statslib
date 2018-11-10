import numpy as np 
from statslib.main.optimization import GradientDescent

def f(x):
    if x > 1: 
        return 5*x
    elif x > 0:
        return x
    else:
        return 0

def df(x):
    if x > 1:
        return 5
    elif x > 0:
        return 1
    else:
        return 0

xinit = 10
optimizer = GradientDescent(0.8, 1e-3, 100)
xfinal = optimizer.run(xinit, f, df) 
print(xfinal)