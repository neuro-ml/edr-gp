import pytest
import numpy as np
from copy import deepcopy
from edrgp.regression import GaussianProcessRegressor
from edrgp.edr import EffectiveDimensionalityReduction
from edrgp.datasets import get_gaussian_inputs, get_tanh_targets
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA


def get_2d_data(mean=None):
    '''Returns a simple 2D dataset'''
    if mean is None:
        mean = [0, 0]
    X = get_gaussian_inputs(
        eig_values=[1, 0.3], sample_size=500,
        eig_vectors=np.array([[1, 1], [-1, 1]]),
        mean=mean)
    y = get_tanh_targets(X, [0.5, 0.5])
    return X, y


@pytest.mark.parametrize("mean", [[0, 0], [10, -10]])
def test_mi(mean):
    X, y = get_2d_data(mean)
    edr = EffectiveDimensionalityReduction(GaussianProcessRegressor(),
                                           PCA(n_components=1), True)
    edr.fit(X, y)
    mi = mutual_info_regression(edr.transform(X), y)[0]
    assert mi > 1


@pytest.mark.parametrize("normalize", [True, False])
def test_translation(normalize):
    X, y = get_2d_data(mean=[10, -10])
    edr = EffectiveDimensionalityReduction(GaussianProcessRegressor(),
                                           PCA(n_components=1), normalize)
    edr.fit(X, y)
    components_shift = edr.components_

    X -= X.mean(0)
    edr2 = deepcopy(edr)
    edr2.fit(X, y)
    components_no_shift = edr2.components_
    assert np.allclose(components_shift, components_no_shift, rtol=1e-3)


@pytest.mark.parametrize("mean", [[0, 0, 0, 0], [10, -10, 100, -100]])
def test_preprocess(mean):
    X = get_gaussian_inputs(
        eig_values=[1, 0.3, 0.001, 0.001], sample_size=500, mean=mean)
    y = get_tanh_targets(X, [0.5, 0.5, 0, 0])

    edr = EffectiveDimensionalityReduction(GaussianProcessRegressor(),
                                           PCA(n_components=1), True,
                                           PCA(n_components=2))
    edr.fit(X, y)
    # print(edr.components_)
    # mi = mutual_info_regression(edr.transform(X), y)[0]
    # assert mi > 1
    components_shift = edr.components_

    X -= X.mean(0)
    edr = EffectiveDimensionalityReduction(GaussianProcessRegressor(),
                                           PCA(n_components=1), True,
                                           PCA(n_components=2))
    edr.fit(X, y)
    components_no_shift = edr.components_
    assert np.allclose(components_shift, components_no_shift, rtol=1e-3)
