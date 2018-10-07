# PCA data decomposition

import numpy as np 

class basePCA(object):
    def __init__(self, expected_var_ratio):
        self.explained_variance_ratio_ = expected_var_ratio

    def fit(self, X):
        n = X.shape[0]
        mean = np.mean(X, axis=0, keepdims=True)
        X -= mean
        covariance = X.T @ X
        U, S, Vt = np.linalg.svd(covariance)
        explained_variance = S / (n - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio = explained_variance / total_var
        var_ratio = 0
        for i, ratio in enumerate(explained_variance_ratio):
            var_ratio += ratio
            if var_ratio > self.explained_variance_ratio_:
                numOf_PCA_components = i+1
                break
        self.transform_matrix_ = Vt[:numOf_PCA_components]
        self.singular_value_ = S[:numOf_PCA_components]
        self.explained_variance_ratio_ = explained_variance_ratio[:numOf_PCA_components]
        return self

    def transform(self, X):
        return (self.transform_matrix_ @ X.T).T 
    
