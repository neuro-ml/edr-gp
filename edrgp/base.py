import numpy as np
from sklearn.utils import check_X_y, assert_all_finite, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import BaseEstimator
from copy import deepcopy
from GPy import kern as gpy_kern
from abc import ABCMeta, abstractmethod
import six


class _BaseGP(six.with_metaclass(ABCMeta, BaseEstimator)):
    def __init__(self, kernels=None, kernel_options=None, Y_metadata=None,
                 mean_function=None):
        self.kernels = kernels
        self.kernel_options = kernel_options
        self.Y_metadata = Y_metadata
        self.mean_function = mean_function
        self.estimator_ = None

    def fit(self, X, y, **opt_kws):
        X, y = self._check_input(X, y)

        kernel = self._make_kernel(X)
        self.estimator_ = self._get_model(X, y, kernel)

        opt_kws.setdefault('messages', False)
        opt_kws.setdefault('max_iters', 1000)
        self.estimator_.optimize(**opt_kws)
        return self

    def _check_input(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        if self._estimator_type == 'classifier':
            check_classification_targets(y)
        y = y[:, np.newaxis]
        return X, y

    def _make_kernel(self, X):
        # kernel will be initiated as 'rbf' in model automatically
        if self.kernels is None:
            return self.kernels

        if isinstance(self.kernels, str):
            self.kernels = [self.kernel]

        kernels = [getattr(gpy_kern, kern) for kern in self.kernels]
        input_dim = {'input_dim': X.shape[1]}

        if self.kernel_options is None:
            options = [input_dim] * len(self.kernels)
        elif len(kernels) == len(self.kernel_options):
            options = deepcopy(self.kernel_options)
            for opt in options:
                opt.update(input_dim)
        else:
            raise(ValueError)

        kernel = np.sum([kern(**opt) for kern, opt in zip(kernels, options)])
        return kernel

    @abstractmethod
    def _get_model(self, X, y, kenrel):
        pass

    def predict(self, X):
        X = check_array(X, accept_sparse=False)
        y_pred = self.estimator_.predict(X)[0][:, 0]
        assert_all_finite(y_pred)
        return y_pred

    def predict_variance(self, X):
        return self.estimator_.predict(X)[1]

    def predict_gradient(self, X):
        return self.estimator_.predictive_gradients(X)[0][:, :, 0]
