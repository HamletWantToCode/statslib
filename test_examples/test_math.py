import numpy as np 
from statslib.tools.utils import euclidean_distance, manhattan_distance, svd_solver

# check distance
X = np.array([[1, 2, 3],
              [4, 5, 6]])

true_eu_dist = np.array([[0, 3*np.sqrt(3)],
                         [3*np.sqrt(3), 0]])

true_manh_dist = np.array([[0, 9],
                           [9, 0]])

compute_eu_dist = euclidean_distance(X, X)
compute_manh_dist = manhattan_distance(X, X)

eu_err = abs(compute_eu_dist - true_eu_dist)
manh_err = abs(compute_manh_dist - true_manh_dist)

# check svd
A = np.array([[0.1, 5, 0.5],
              [5, 1, 20],
              [0.5, 20, 9]])
x = np.ones(3)[:, np.newaxis]
b = A @ x
compute_x = svd_solver(A, b)
x_err = abs(compute_x - x)

print(eu_err)
print(manh_err)
print(x_err)

