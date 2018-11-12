import numpy as np 
from statslib.main.optimization import *

def f(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

def df(x):
    return np.array([-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)]) 

# optimizer = GradientDescent(0.005, 1e-3, 500)
optimizer = NesterovGD(0.005, 1e-3, 100, 0.5)
xinit = np.array([0, 1.0])
xfinal = optimizer.run(xinit, f, df)
print(xfinal)

