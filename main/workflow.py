# machine learning workflow

import numpy as np 

class Workflow(object):
    def __init__(self, n_components, gamma, lambda_, kernel, kernel_gd, model):
        self.n_components_ = n_components
        self.kernel = kernel(gamma)
        self.kernel_gd = kernel_gd(gamma)
        self.model = model(self.kernel, lambda_)

    def transform(self, X):
        n_ = X.shape[0]
        mean_X = np.mean(X, axis=0)
        Cov = (X - mean_X).T @ (X - mean_X) / n_
        U, _, _ = np.linalg.svd(Cov)
        self.mean_X = mean_X
        self.tr_mat_ = U[:, :self.n_components_]
        return self

    def fit(self, X, y, dy=None):
        self.transform(X)
        X_t = (X-self.mean_X) @ self.tr_mat_
        self.model.fit(X_t, y)
        return self

    def predict(self, X):
        n_, D_ = X.shape
        X_t = (X-self.mean_X) @ self.tr_mat_
        pred_y = self.model.predict(X_t)
        K_gd = self.kernel_gd(X_t, self.model.X_fit_)
        pred_dyt = (D_-1)*(K_gd @ self.model.coef_).reshape((n_, -1))
        pred_dy = pred_dyt @ self.tr_mat_.T 
        return pred_y, pred_dy