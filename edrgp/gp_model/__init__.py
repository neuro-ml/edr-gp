from .classification import (GaussianProcessClassifier,
                             SparseGaussianProcessClassifier)
from .regression import (GaussianProcessRegressor,
                         SparseGaussianProcessRegressor)

__all__ = ['GaussianProcessClassifier',
           'GaussianProcessRegressor',
           'SparseGaussianProcessClassifier',
           'SparseGaussianProcessRegressor']
