import pytest
import numpy as np
from edrgp.regression import GaussianProcessRegressor
from edrgp.edr import EffectiveDimensionalityReduction
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA


def get_data(sample_size=500, noise_std=0.05):
    # generate covariance mat
    U = np.array([[1, 1], [-1, 1]])
    S = np.diag([1, 0.3])
    cov = np.dot(np.dot(U, S), U.T)
    # generate centered inputs
    X = np.random.multivariate_normal([0, 0], cov, 500)
    X -= X.mean(0)
    y = func(X) + noise_std * np.random.randn(sample_size)
    return X, y


def func(X):
    return np.tanh((X[:, 0] + X[:, 1]) * 0.5)


def test_smoke():
    X, y = get_data()
    edr = EffectiveDimensionalityReduction(GaussianProcessRegressor(),
                                           PCA(n_components=1), True)
    edr.fit(X, y)
    mi = mutual_info_regression(edr.transform(X), y)[0]
    assert mi > 2
