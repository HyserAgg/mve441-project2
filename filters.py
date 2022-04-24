import numpy as np


from scipy.stats.stats import pearsonr

# Scikit-learn stuff
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, RobustScaler

class VarianceFilter(BaseEstimator, TransformerMixin):
    def __init__(self, thrshld):
        self.thrshld = thrshld

    def fit(self, X, y = None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_norm = RobustScaler().fit_transform(X.copy())
        X_var = X_norm.var(axis=0)
        X_ = X_copy[:,(X_var < self.thrshld)]
        _,m = np.shape(X_)
        print(f"VF returns features of dim {m}")
        return X_
    
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, thrshld):
        self.thrshld = thrshld

    def fit(self, X, y = None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        i = 0
        _,m = np.shape(X_)
        while i<m: 
            _, m = np.shape(X_)
            corr = np.zeros(shape = m, dtype = bool)
            # Find correlation with all columns j>i
            for j in range(i+1,m):
                c, _ = pearsonr(X_[:,i],X_[:,j])
                corr[j] = np.abs(c) > self.thrshld

            X_ = np.delete(X_, corr, axis = 1)
            if np.mod(m-i,50)== 0:
                print(i,m)
            i += 1
        return X_