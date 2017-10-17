import pytest
from edrgp.regression import GaussianProcessRegressor
from edrgp.edr import EffectiveDimensionalityReduction
from edrgp.datasets import get_2d_data
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA


def test_smoke():
    X, y = get_2d_data()
    edr = EffectiveDimensionalityReduction(GaussianProcessRegressor(),
                                           PCA(n_components=1), True)
    edr.fit(X, y)
    mi = mutual_info_regression(edr.transform(X), y)[0]
    assert mi > 2
